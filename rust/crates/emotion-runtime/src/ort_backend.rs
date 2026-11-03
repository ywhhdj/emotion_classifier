//! onnxruntime 后端（feature `backend-ort`）。
//!
//! 直接复用训练/导出阶段产出的整图 ONNX（input_ids / attention_mask -> logits），
//! 无需在 Rust 侧复现 XLM-R 前向，适合服务端 GPU / 高吞吐场景。

use crate::{Backend, EngineConfig};
use emotion_core::{Error as CoreError, Result as CoreResult};
use std::path::Path;

pub struct OrtBackend {
    session: ort::Session,
    num_labels: usize,
    hidden: usize,
}

impl OrtBackend {
    pub fn load(cfg: &EngineConfig) -> anyhow::Result<Self> {
        let dir = Path::new(&cfg.model_dir);
        let model = ["emotion_classifier.onnx", "emotion_classifier_fp16.onnx",
                     "emotion_classifier_int8.onnx"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.exists())
            .ok_or_else(|| anyhow::anyhow!("未找到 ONNX 模型: {}", dir.display()))?;

        let session = ort::Session::builder()?
            .with_intra_threads(cfg.num_threads)?
            .with_inter_threads(1)?
            .commit_from_file(&model)?;

        let num_labels = session
            .outputs()
            .first()
            .and_then(|o| o.output_type.tensor_dimensions().ok())
            .and_then(|d| d.last().map(|x| *x as usize))
            .unwrap_or(18);
        let hidden = crate::XlmrConfig::from_json(dir)
            .map(|c| c.hidden)
            .unwrap_or(768);

        Ok(Self { session, num_labels, hidden })
    }
}

impl Backend for OrtBackend {
    fn forward(&self, ids: &[Vec<u32>], mask: &[Vec<u32>]) -> CoreResult<Vec<Vec<f32>>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let (b, n) = (ids.len(), ids[0].len());
        let flat: Vec<i64> = ids.iter().flatten().map(|&x| x as i64).collect();
        let mflat: Vec<i64> = mask.iter().flatten().map(|&x| x as i64).collect();

        let ids_t = ort::value::Tensor::from_array(([b, n], flat))
            .map_err(|e| CoreError::Forward(e.to_string()))?;
        let mask_t = ort::value::Tensor::from_array(([b, n], mflat))
            .map_err(|e| CoreError::Forward(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => ids_t,
                "attention_mask" => mask_t
            ])
            .map_err(|e| CoreError::Forward(e.to_string()))?;

        let (shape, data) = outputs["logits"]
            .try_extract_raw_tensor::<f32>()
            .map_err(|e| CoreError::Forward(e.to_string()))?;
        let rows = shape.first().copied().unwrap_or(b as i64) as usize;
        let cols = if self.num_labels > 0 { self.num_labels } else { data.len() / rows.max(1) };
        Ok(data.chunks(cols).map(|c| c.to_vec()).take(rows).collect())
    }

    fn hidden_size(&self) -> usize {
        self.hidden
    }

    fn num_labels(&self) -> usize {
        self.num_labels
    }
}
