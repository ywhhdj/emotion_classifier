//! XLM-RoBERTa 序列分类前向（candle 实现）。
//!
//! 两个易错点：
//!   1. position_ids 从 padding_idx + 1 = 2 起算（见 `create_position_ids`）；
//!   2. XLM-R **没有** token_type_ids（不做 NSP），前向不需要该输入。

use crate::{create_position_ids, Backend, EngineConfig, XlmrConfig};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{embedding, linear, Activation, Embedding, Linear, VarBuilder};
use emotion_core::{Error as CoreError, Result as CoreResult};
use std::path::Path;

struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl SelfAttention {
    fn forward(&self, x: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let q = self.q.forward(x)?;
        let k = self.k.forward(x)?;
        let v = self.v.forward(x)?;

        let q = q.reshape((b, n, self.heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, n, self.heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, n, self.heads, self.head_dim))?.transpose(1, 2)?;

        let mut scores = (q.matmul(&k.t()?)? * self.scale)?;
        scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
        let ctx = probs.matmul(&v)?;
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, n, self.heads * self.head_dim))?;
        self.out.forward(&ctx)
    }
}

struct Layer {
    attn: SelfAttention,
    attn_ln: candle_nn::LayerNorm,
    ff1: Linear,
    ff2: Linear,
    ff_ln: candle_nn::LayerNorm,
}

impl Layer {
    fn forward(&self, x: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.attn.forward(x, mask)?;
        let x = (x + h)?;
        let x = self.attn_ln.forward(&x)?;
        let h = self.ff2.forward(&self.ff1.forward(&x)?.gelu()?)?;
        let x = (x + h)?;
        self.ff_ln.forward(&x)
    }
}

pub struct XlmrForClassification {
    word_emb: Embedding,
    pos_emb: Embedding,
    emb_ln: candle_nn::LayerNorm,
    layers: Vec<Layer>,
    pooler: Linear,
    classifier: Linear,
    cfg: XlmrConfig,
    device: Device,
}

impl XlmrForClassification {
    pub fn load(engine_cfg: &EngineConfig) -> anyhow::Result<Self> {
        let dir = Path::new(&engine_cfg.model_dir);
        let cfg = XlmrConfig::from_json(dir).unwrap_or_default();
        let device = Device::Cpu;
        let dtype = crate::dtype_of(engine_cfg.precision);

        // safetensors 走 mmap：加载快、多实例共享同一份物理内存
        let weight_path = ["model.safetensors", "pytorch_model.bin"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.exists())
            .ok_or_else(|| anyhow::anyhow!("未找到 model.safetensors / pytorch_model.bin in {:?}", dir))?;

        let tensors = if weight_path.extension().map(|e| e == "safetensors").unwrap_or(false) {
            unsafe { candle_core::safetensors::MmapedSafetensors::new(&weight_path)? }
                .tensors()
                .into_iter()
                .map(|(k, v)| (k, v.to_tensor()))
                .collect::<std::collections::HashMap<String, candle_core::Tensor>>()
        } else {
            candle_core::safetensors::load(&weight_path, &device)?
        };
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);

        let word_emb = embedding(cfg.vocab_size, cfg.hidden,
            vb.pp("roberta.embeddings.word_embeddings"))?;
        let pos_emb = embedding(cfg.max_pos, cfg.hidden,
            vb.pp("roberta.embeddings.position_embeddings"))?;
        let emb_ln = candle_nn::layer_norm(cfg.hidden, cfg.eps,
            vb.pp("roberta.embeddings.LayerNorm"))?;

        let head_dim = cfg.hidden / cfg.heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = vb.pp(format!("roberta.encoder.layer.{i}"));
            let attn = SelfAttention {
                q: linear(cfg.hidden, cfg.hidden, p.pp("attention.self.query"))?,
                k: linear(cfg.hidden, cfg.hidden, p.pp("attention.self.key"))?,
                v: linear(cfg.hidden, cfg.hidden, p.pp("attention.self.value"))?,
                out: linear(cfg.hidden, cfg.hidden, p.pp("attention.output.dense"))?,
                heads: cfg.heads,
                head_dim,
                scale,
            };
            layers.push(Layer {
                attn,
                attn_ln: candle_nn::layer_norm(cfg.hidden, cfg.eps, p.pp("attention.output.LayerNorm"))?,
                ff1: linear(cfg.hidden, cfg.intermediate, p.pp("intermediate.dense"))?,
                ff2: linear(cfg.intermediate, cfg.hidden, p.pp("output.dense"))?,
                ff_ln: candle_nn::layer_norm(cfg.hidden, cfg.eps, p.pp("output.LayerNorm"))?,
            });
        }

        let pooler = linear(cfg.hidden, cfg.hidden, vb.pp("roberta.pooler.dense"))?;
        let classifier = linear(cfg.hidden, cfg.num_labels, vb.pp("classifier"))?;

        Ok(Self { word_emb, pos_emb, emb_ln, layers, pooler, classifier, cfg, device })
    }

    /// attention mask -> 加性 mask（1 的位置填 -inf）
    fn build_mask(&self, mask: &[Vec<u32>], n: usize) -> candle_core::Result<Tensor> {
        let b = mask.len();
        let mut v = vec![0f32; b * n];
        for i in 0..b {
            for j in 0..n {
                if mask[i][j] == 0 {
                    v[i * n + j] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor::from_vec(v, (b, n), &self.device)?
            .unsqueeze(1)?
            .unsqueeze(2)?
            .expand((b, self.cfg.heads, n, n))
    }
}

impl Backend for XlmrForClassification {
    fn forward(&self, ids: &[Vec<u32>], mask: &[Vec<u32>]) -> CoreResult<Vec<Vec<f32>>> {
        let (b, n) = (ids.len(), ids.first().map(|r| r.len()).unwrap_or(0));
        if b == 0 || n == 0 {
            return Ok(Vec::new());
        }
        let flat: Vec<u32> = ids.iter().flatten().copied().collect();
        let input_ids = Tensor::from_vec(flat, (b, n), &self.device)
            .map_err(|e| CoreError::Forward(e.to_string()))?;

        let pos = create_position_ids(ids, 1);
        let pos_flat: Vec<u32> = pos.iter().flatten().copied().collect();
        let pos_ids = Tensor::from_vec(pos_flat, (b, n), &self.device)
            .map_err(|e| CoreError::Forward(e.to_string()))?;

        let attn_mask = self.build_mask(mask, n).map_err(|e| CoreError::Forward(e.to_string()))?;

        let mut h = self.word_emb.forward(&input_ids)
            .map_err(|e| CoreError::Forward(e.to_string()))?;
        let p = self.pos_emb.forward(&pos_ids).map_err(|e| CoreError::Forward(e.to_string()))?;
        h = (h + p).map_err(|e| CoreError::Forward(e.to_string()))?;
        h = self.emb_ln.forward(&h).map_err(|e| CoreError::Forward(e.to_string()))?;

        for layer in &self.layers {
            h = layer.forward(&h, &attn_mask).map_err(|e| CoreError::Forward(e.to_string()))?;
        }

        // [CLS]（XLM-R 的 <s>，下标 0）-> pooler -> tanh -> 分类头
        let cls = h.i((.., 0, ..)).map_err(|e| CoreError::Forward(e.to_string()))?;
        let pooled = self.pooler.forward(&cls)
            .map_err(|e| CoreError::Forward(e.to_string()))?
            .tanh()
            .map_err(|e| CoreError::Forward(e.to_string()))?;
        let logits = self.classifier.forward(&pooled)
            .map_err(|e| CoreError::Forward(e.to_string()))?
            .to_dtype(DType::F32)
            .map_err(|e| CoreError::Forward(e.to_string()))?;

        let v: Vec<f32> = logits.flatten_all()
            .map_err(|e| CoreError::Forward(e.to_string()))?
            .to_vec1()
            .map_err(|e| CoreError::Forward(e.to_string()))?;
        let k = self.cfg.num_labels;
        Ok(v.chunks(k).map(|c| c.to_vec()).collect())
    }

    fn hidden_size(&self) -> usize {
        self.cfg.hidden
    }

    fn num_labels(&self) -> usize {
        self.cfg.num_labels
    }
}
