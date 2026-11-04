//! onnxruntime 后端（feature `backend-ort`，v0.2.0 默认）。

use crate::{Backend, EngineConfig, XlmrConfig};
use candle_core::IntDType;
use emotion_core::{model_filename, Error as CoreError, Result as CoreResult, MODEL_FILES};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::path::PathBuf;
use std::sync::Mutex;

pub struct OrtBackend {
    session: Mutex<Session>,
    output_name: String,
    num_labels: usize,
    hidden: usize,
}

impl OrtBackend {
    pub fn load(cfg: &EngineConfig) -> anyhow::Result<Self> {
        let dir = &cfg.model_dir;
        let fallback: Vec<PathBuf> = MODEL_FILES
            .iter()
            .map(|(_, f)| dir.join(*f))
            .collect();

        let model: Option<PathBuf> = if let Some(p) = &cfg.model_path {
            Some(p.clone())
        } else {
            let preferred = dir.join(model_filename(cfg.precision));
            if preferred.exists() {
                Some(preferred)
            } else {
                fallback.iter().find(|p| p.exists()).cloned()
            }
        };

        // rc.13+：SessionBuilder 的错误类型不实现 Send + Sync，
        // 不能直接 `?` 成 anyhow::Error；必须显式转成 String 或 ort::Error<()>。
        // 同时 Session::builder 需要 &Environment，用 ort::init() 拿全局环境。
         let env= ort::init()
            .with_name("emotion_classifier")
            .build()
            .map_err(|e| anyhow::anyhow!("ORT 环境初始化失败: {e}"))?;

        // Session::builder 不再需要显式传入 environment 参数
        // rc.15 会自动使用已经 commit 的全局环境
        let mut builder = Session::builder(&env)
            .map_err(|e| anyhow::anyhow!("创建 SessionBuilder 失败: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| anyhow::anyhow!("设置优化级别失败: {e}"))?
            .with_intra_threads(cfg.num_threads)
            .map_err(|e| anyhow::anyhow!("设置 intra_threads 失败: {e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("设置 inter_threads 失败: {e}"))?;


        let session = match cfg.model_bytes.as_deref() {
            Some(bytes) => builder
                .commit_from_memory(bytes)
                .map_err(|e| anyhow::anyhow!("从内存加载 ONNX 失败: {e}"))?,
            None => {
                let path = model.ok_or_else(|| {
                    anyhow::anyhow!(
                        "未找到三档 ONNX 模型（{}），目录: {}",
                        model_filename(cfg.precision),
                        dir.display()
                    )
                })?;
                builder
                    .commit_from_file(&path)
                    .map_err(|e| anyhow::anyhow!("从文件加载 ONNX 失败: {e}"))?
            }
        };

        // rc.15：inputs/outputs 是「方法」而不是「字段」，
        // 返回 &[Outlet]，Outlet 的 name 是方法 .name() 或字段 .name。
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        if !input_names.iter().any(|n| n == "input_ids")
            || !input_names.iter().any(|n| n == "attention_mask")
        {
            anyhow::bail!(
                "ONNX 整图需要 input_ids/attention_mask 两个输入，实际为 {:?}",
                input_names
            );
        }

        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| "logits".to_string());

        let num_labels = session
            .outputs()
            .first()
            .and_then(|o| o.dtype().tensor_shape())
            .and_then(|d| d.last().copied())
            .filter(|&n| n > 0)
            .unwrap_or(18).as_usize();

        let hidden = XlmrConfig::from_json(dir).map(|c| c.hidden).unwrap_or(768);

        Ok(Self {
            session: Mutex::new(session),
            output_name,
            num_labels,
            hidden,
        })
    }
}

impl Backend for OrtBackend {
    fn forward(&self, ids: &[Vec<u32>], mask: &[Vec<u32>]) -> CoreResult<Vec<Vec<f32>>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let (b, n) = (ids.len(), ids[0].len());
        if n == 0
            || ids.iter().any(|r| r.len() != n)
            || mask.len() != b
            || mask.iter().any(|r| r.len() != n)
        {
            return Err(CoreError::InvalidInput(
                "ids/mask 必须是等长的二维 batch（由 Engine::pad_batch 保证）".into(),
            ));
        }
        let flat: Vec<i64> = ids.iter().flatten().map(|&x| x as i64).collect();
        let mflat: Vec<i64> = mask.iter().flatten().map(|&x| x as i64).collect();

        let ids_t = ort::value::Tensor::from_array(([b, n], flat))
            .map_err(|e| CoreError::Forward(e.to_string()))?;
        let mask_t = ort::value::Tensor::from_array(([b, n], mflat))
            .map_err(|e| CoreError::Forward(e.to_string()))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| CoreError::Forward(format!("session lock failed: {e}")))?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_t,
                "attention_mask" => mask_t
            ])
            .map_err(|e| CoreError::Forward(e.to_string()))?;

        // rc.11+：try_extract_raw_tensor 已重命名为 try_extract_tensor，
        // 返回 (&Shape, &[f32])，而不是 (Vec<i64>, &[f32])。
        let (shape, data) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| CoreError::Forward(e.to_string()))?;

        let rows = shape
            .first()
            .copied()
            .unwrap_or(b as i64)
            .max(1) as usize;
        let cols = if self.num_labels > 0 {
            self.num_labels
        } else {
            (data.len() / rows).max(1)
        };
        Ok(data.chunks(cols).map(|c| c.to_vec()).take(rows).collect())
    }

    fn hidden_size(&self) -> usize {
        self.hidden
    }

    fn num_labels(&self) -> usize {
        self.num_labels
    }
}