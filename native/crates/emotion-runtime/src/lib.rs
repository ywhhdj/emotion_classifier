//! 推理后端与批调度（v0.2.0）。
//!
//! * `ort`（默认）：直接加载 Release 的三档 ONNX 整图；
//! * `candle`（可选）：纯 Rust 前向，需 model.safetensors。
//!
//! 进程内**只有一个** `Engine`：`global_engine()` 用配置指纹校验，
//! 相同配置复用 `Arc`，不同配置直接报错，避免重复初始化模型。
//! 并发只读推理通过 `Arc` 共享，无需全局锁，不牺牲吞吐。

use candle_core::DType;
use emotion_core::{
    BackendKind, EngineConfig, Error as CoreError, Precision, Result as CoreResult,
};
use emotion_tokenizer::{Encoded, Tokenizer};
use std::path::Path;
use std::sync::{Arc, Mutex};

pub mod batcher;
pub mod candle_xlmr;
#[cfg(feature = "backend-ort")]
pub mod ort_backend;

pub use candle_xlmr::XlmrForClassification;

static GLOBAL_ENGINE: Mutex<Option<Arc<Engine>>> = Mutex::new(None);

/// 获取进程内唯一引擎。配置不同时报错（调用 `release_global_engine()` 才能换档）。
pub fn global_engine(cfg: EngineConfig) -> CoreResult<Arc<Engine>> {
    let mut guard = GLOBAL_ENGINE
        .lock()
        .map_err(|e| CoreError::ModelLoad(format!("单例锁异常: {e}")))?;
    if let Some(existing) = guard.as_ref() {
        if existing.config_key() == cfg.key() {
            return Ok(Arc::clone(existing));
        }
        return Err(CoreError::ModelLoad(format!(
            "进程内只允许一个 Rust 引擎实例；已存在 [{}]，请求 [{}]。\
             如需切换请先调用 release_global_engine()",
            existing.config_key(),
            cfg.key()
        )));
    }
    let engine = Arc::new(Engine::load(cfg)?);
    *guard = Some(Arc::clone(&engine));
    Ok(engine)
}

/// 释放唯一引擎（配合 Python 侧 `release_model()`）。
pub fn release_global_engine() -> bool {
    match GLOBAL_ENGINE.lock() {
        Ok(mut g) => {
            let had = g.is_some();
            *g = None;
            had
        }
        Err(_) => false,
    }
}

/// 当前是否已有实例（不触发加载）。
pub fn has_global_engine() -> bool {
    GLOBAL_ENGINE.lock().map(|g| g.is_some()).unwrap_or(false)
}

/// 统一的推理后端接口。
pub trait Backend: Send + Sync {
    fn forward(&self, ids: &[Vec<u32>], mask: &[Vec<u32>]) -> CoreResult<Vec<Vec<f32>>>;
    fn hidden_size(&self) -> usize;
    fn num_labels(&self) -> usize;
}

/// 打包后的引擎：分词 + 后端 + 动态 padding 批调度。
pub struct Engine {
    tokenizer: Tokenizer,
    backend: Box<dyn Backend>,
    cfg: EngineConfig,
}

impl Engine {
    pub fn load(cfg: EngineConfig) -> CoreResult<Self> {
        // 先把默认目录物化为具名变量，生命周期覆盖整个函数
        let default_tokenizer_dir = cfg.model_dir.join("tokenizer");
        let tokenizer_dir = cfg
            .tokenizer_dir
            .as_ref()
            .unwrap_or(&default_tokenizer_dir);
        let tokenizer = Tokenizer::from_dir(tokenizer_dir, cfg.max_seq_len, cfg.use_lang_token)
            .map_err(|e| CoreError::Tokenizer(e.to_string()))?;

        let backend: Box<dyn Backend> = match cfg.backend {
            BackendKind::Candle => Box::new(
                XlmrForClassification::load(&cfg)
                    .map_err(|e| CoreError::ModelLoad(e.to_string()))?,
            ),
            #[cfg(feature = "backend-ort")]
            BackendKind::Onnx => Box::new(
                ort_backend::OrtBackend::load(&cfg)
                    .map_err(|e| CoreError::ModelLoad(e.to_string()))?,
            ),
            #[cfg(not(feature = "backend-ort"))]
            BackendKind::Onnx => {
                return Err(CoreError::ModelLoad(
                    "未编译 backend-ort feature，无法加载三档 ONNX；\
                     请用 candle 后端或重新构建 wheel".into(),
                ))
            }
        };

        Ok(Self { tokenizer, backend, cfg })
    }

    pub fn config_key(&self) -> String {
        self.cfg.key()
    }

    pub fn hidden_size(&self) -> usize {
        self.backend.hidden_size()
    }

    pub fn num_labels(&self) -> usize {
        self.backend.num_labels()
    }

    /// 按 batch 内最长动态 padding（与训练侧 `make_collate` 一致）。
    /// v0.1 把每条都 pad 到固定 48/64，极短文本场景下浪费大量算力。
    fn pad_batch(&self, ids: &[Vec<u32>], mask: &[Vec<u32>])
                 -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let pad = self.tokenizer.pad_id;
        let n = ids.iter().map(|r| r.len()).max().unwrap_or(0);
        let mut p_ids = Vec::with_capacity(ids.len());
        let mut p_mask = Vec::with_capacity(mask.len());
        for (row, m) in ids.iter().zip(mask.iter()) {
            let mut r = row.clone();
            let mut mm = m.clone();
            r.resize(n, pad);
            mm.resize(n, 0);
            p_ids.push(r);
            p_mask.push(mm);
        }
        (p_ids, p_mask)
    }

    /// 文本 -> logits（分词 + 长度分桶 + 动态 padding + 前向，全在内核完成）。
    pub fn logits(&self, texts: &[String], langs: &[String]) -> CoreResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if langs.len() != texts.len() {
            return Err(CoreError::InvalidInput("lang 数量与文本数量不一致".into()));
        }

        let encoded: Vec<Encoded> = self.tokenizer.encode_batch(texts, langs);
        let mut ids = Vec::with_capacity(encoded.len());
        let mut mask = Vec::with_capacity(encoded.len());
        for e in encoded {
            ids.push(e.ids);
            mask.push(e.mask);
        }

        // 长度分桶：批内按 token 数排序，减少 padding 浪费
        let mut order: Vec<usize> = (0..ids.len()).collect();
        order.sort_by_key(|&i| mask[i].iter().sum::<u32>());

        let mut out: Vec<Vec<f32>> = vec![Vec::new(); ids.len()];
        for chunk in order.chunks(self.cfg.max_batch.max(1)) {
            let b_ids: Vec<Vec<u32>> = chunk.iter().map(|&i| ids[i].clone()).collect();
            let b_mask: Vec<Vec<u32>> = chunk.iter().map(|&i| mask[i].clone()).collect();
            let (p_ids, p_mask) = self.pad_batch(&b_ids, &b_mask);
            let logits = self.backend.forward(&p_ids, &p_mask)?;
            for (j, &i) in chunk.iter().enumerate() {
                out[i] = logits[j].clone();
            }
        }
        Ok(out)
    }

    /// 已分词输入直入（Python 侧 numpy 零拷贝走这条路径）。
    pub fn logits_from_ids(&self, ids: &[Vec<u32>], mask: &[Vec<u32>])
                          -> CoreResult<Vec<Vec<f32>>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        if ids.len() != mask.len() {
            return Err(CoreError::InvalidInput("ids 与 mask 行数不一致".into()));
        }
        let step = self.cfg.max_batch.max(1);
        let mut out = Vec::with_capacity(ids.len());
        // v0.1 bug：mask 永远取前 chunk.len() 行，第二批开始全部错位
        for (ids_chunk, mask_chunk) in ids.chunks(step).zip(mask.chunks(step)) {
            let (p_ids, p_mask) = self.pad_batch(ids_chunk, mask_chunk);
            let logits = self.backend.forward(&p_ids, &p_mask)?;
            out.extend(logits);
        }
        Ok(out)
    }

    pub fn close(&self) {}
}

/// 从 config.json 读取 XLM-R 结构参数（candle 后端用）。
pub struct XlmrConfig {
    pub vocab_size: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub max_pos: usize,
    pub num_labels: usize,
    pub eps: f64,
}

impl Default for XlmrConfig {
    fn default() -> Self {
        Self {
            vocab_size: 250_002,
            hidden: 768,
            layers: 12,
            heads: 12,
            intermediate: 3072,
            max_pos: 514,
            num_labels: 18,
            eps: 1e-5,
        }
    }
}

impl XlmrConfig {
    pub fn from_json(dir: &Path) -> anyhow::Result<Self> {
        let p = dir.join("config.json");
        let s = std::fs::read_to_string(&p)?;
        let v: serde_json::Value = serde_json::from_str(&s)?;
        let g = |k: &str, d: usize| -> usize {
            v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(d)
        };
        let d = Self::default();
        Ok(Self {
            vocab_size: g("vocab_size", d.vocab_size),
            hidden: g("hidden_size", d.hidden),
            layers: g("num_hidden_layers", d.layers),
            heads: g("num_attention_heads", d.heads),
            intermediate: g("intermediate_size", d.intermediate),
            max_pos: g("max_position_embeddings", d.max_pos),
            num_labels: v
                .get("num_labels")
                .or_else(|| v.get("id2label"))
                .and_then(|val| {
                    // 如果是数字直接取，如果是对象取长度
                    val.as_u64().or_else(|| val.as_object().map(|m| m.len() as u64))
                })
                .map(|x| x as usize)
                .unwrap_or(d.num_labels),
            eps: v.get("layer_norm_eps").and_then(|x| x.as_f64()).unwrap_or(d.eps),
        })
    }
}

pub fn dtype_of(p: Precision) -> DType {
    match p {
        Precision::Fp32 => DType::F32,
        Precision::Fp16 => DType::F16,
        Precision::Int8 => DType::F32, // int8 走量化张量，计算上仍以 f32 还原
    }
}

/// 复现 HF `create_position_ids_from_input_ids`（candle 后端用）。
///
/// XLM-R 的 position embedding 从 `padding_idx + 1 = 2` 起算。
pub fn create_position_ids(ids: &[Vec<u32>], pad_id: u32) -> Vec<Vec<u32>> {
    ids.iter()
        .map(|row| {
            let mut acc = 0u32;
            row.iter()
                .map(|&t| {
                    if t == pad_id {
                        pad_id
                    } else {
                        acc += 1;
                        acc + pad_id
                    }
                })
                .collect()
        })
        .collect()
}
