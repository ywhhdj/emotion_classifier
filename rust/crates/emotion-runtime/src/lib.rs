//! 推理后端与批调度。
//!
//! 两个后端通过 `trait Backend` 抽象，Python 侧完全无感：
//!   * `candle`（默认）：纯 Rust，无 C 依赖，易交叉编译
//!   * `ort`（feature `backend-ort`）：onnxruntime，服务端 GPU / 高吞吐

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};
use emotion_core::{BackendKind, EngineConfig, Precision, Result as CoreResult};
use emotion_core::Error as CoreError;
use emotion_tokenizer::{Encoded, Tokenizer};
use std::collections::HashMap;
use std::path::Path;

pub mod batcher;
pub mod candle_xlmr;
#[cfg(feature = "backend-ort")]
pub mod ort_backend;

pub use candle_xlmr::XlmrForClassification;

/// 统一的推理后端接口。
pub trait Backend: Send + Sync {
    fn forward(&self, ids: &[Vec<u32>], mask: &[Vec<u32>]) -> CoreResult<Vec<Vec<f32>>>;
    fn hidden_size(&self) -> usize;
    fn num_labels(&self) -> usize;
}

/// 打包后的引擎：分词 + 后端 + 批调度。
pub struct Engine {
    tokenizer: Tokenizer,
    backend: Box<dyn Backend>,
    cfg: EngineConfig,
}

impl Engine {
    pub fn load(cfg: EngineConfig) -> CoreResult<Self> {
        let tokenizer = Tokenizer::from_dir(&cfg.model_dir, cfg.max_seq_len, cfg.use_lang_token)
            .map_err(|e| CoreError::Tokenizer(e.to_string()))?;

        let backend: Box<dyn Backend> = match cfg.backend {
            BackendKind::Candle => {
                Box::new(XlmrForClassification::load(&cfg).map_err(|e| {
                    CoreError::ModelLoad(e.to_string())
                })?)
            }
            #[cfg(feature = "backend-ort")]
            BackendKind::Onnx => Box::new(ort_backend::OrtBackend::load(&cfg)
                .map_err(|e| CoreError::ModelLoad(e.to_string()))?),
            #[cfg(not(feature = "backend-ort"))]
            BackendKind::Onnx => return Err(CoreError::ModelLoad(
                "未编译 backend-ort feature，请使用 candle 后端".into())),
        };

        Ok(Self { tokenizer, backend, cfg })
    }

    pub fn hidden_size(&self) -> usize {
        self.backend.hidden_size()
    }

    pub fn num_labels(&self) -> usize {
        self.backend.num_labels()
    }

    /// 文本 -> logits（分词 + 批调度 + 前向 全在内核完成）。
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
            let logits = self.backend.forward(&b_ids, &b_mask)?;
            for (j, &i) in chunk.iter().enumerate() {
                out[i] = logits[j].clone();
            }
        }
        Ok(out)
    }

    /// 已分词输入直入（Python 侧零拷贝 numpy 走这条路径）。
    pub fn logits_from_ids(&self, ids: &[Vec<u32>], mask: &[Vec<u32>]) -> CoreResult<Vec<Vec<f32>>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(self.cfg.max_batch.max(1)) {
            let m_chunk = &mask[..chunk.len()];
            let logits = self.backend.forward(chunk, m_chunk)?;
            out.extend(logits);
        }
        Ok(out)
    }

    pub fn close(&self) {}
}

/// 从 config.json 读取 XLM-R 结构参数。
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
                .or_else(|| v.get("id2label").and_then(|m| m.as_object()).map(|m| m.len() as u64))
                .and_then(|x| x.as_u64())
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

/// 复现 HF `create_position_ids_from_input_ids`。
///
/// XLM-R 的 position embedding 从 `padding_idx + 1 = 2` 起算；
/// 自实现前向必须复现该逻辑，否则右 padding 的短句位置编码全错。
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
