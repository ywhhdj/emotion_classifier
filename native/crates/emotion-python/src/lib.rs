//! PyO3 绑定 —— 编译为 `emotion_classifier._core`（v0.2.0）。
//!
//! * 进程内单一实例：`emotion_runtime::global_engine()` 用配置指纹校验，
//!   相同配置复用 `Arc<Engine>`，不会重复初始化模型；
//! * 支持传入已加载模型：`model=<onnx bytes>` 走 `commit_from_memory`；
//! * `py.allow_threads(...)` 释放 GIL，多线程只读推理真正并行；
//! * numpy 零拷贝：`PyReadonlyArray2` 进、`PyArray2` 出。

use emotion_core::{
    detect_lang, normalize_lang, BackendKind, Calibration, EngineConfig, Precision,
};
use emotion_postprocess::{probs, top_k as top_k_scores};
use emotion_runtime::{global_engine, release_global_engine, Engine};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

fn to_py_err(e: emotion_core::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn parse_backend(s: &str) -> PyResult<BackendKind> {
    match s.to_lowercase().as_str() {
        "candle" => Ok(BackendKind::Candle),
        "onnx" | "ort" => Ok(BackendKind::Onnx),
        other => Err(PyValueError::new_err(format!("未知后端: {other}"))),
    }
}

fn parse_precision(s: &str) -> PyResult<Precision> {
    match s.to_lowercase().as_str() {
        "fp32" | "float32" => Ok(Precision::Fp32),
        "fp16" | "float16" => Ok(Precision::Fp16),
        "int8" | "quant" => Ok(Precision::Int8),
        other => Err(PyValueError::new_err(format!("未知精度: {other}"))),
    }
}

fn fingerprint(bytes: &[u8]) -> String {
    let mut h = DefaultHasher::new();
    bytes.len().hash(&mut h);
    bytes.iter().take(64).for_each(|b| b.hash(&mut h));
    format!("mem-{:016x}", h.finish())
}

fn build_config(
    model_dir: &str,
    tokenizer_dir: Option<&str>,
    num_threads: usize,
    max_batch: usize,
    max_wait_ms: u64,
    backend: &str,
    precision: &str,
    max_seq_len: usize,
    use_lang_token: bool,
    model_path: Option<String>,
    model: Option<&[u8]>,
) -> PyResult<EngineConfig> {
    let mut cfg = EngineConfig {
        model_dir: std::path::PathBuf::from(model_dir),
        tokenizer_dir: tokenizer_dir.map(std::path::PathBuf::from),
        num_threads,
        max_batch,
        max_wait_ms,
        backend: parse_backend(backend)?,
        precision: parse_precision(precision)?,
        max_seq_len,
        use_lang_token,
        ..EngineConfig::default()
    };
    cfg.model_path = model_path.map(std::path::PathBuf::from);
    if let Some(bytes) = model {
        if bytes.is_empty() {
            return Err(PyValueError::new_err("model 字节为空"));
        }
        cfg.fingerprint = Some(fingerprint(bytes));
        cfg.model_bytes = Some(bytes.to_vec());
    }
    Ok(cfg)
}

fn load_global(cfg: EngineConfig) -> PyResult<Arc<Engine>> {
    global_engine(cfg).map_err(to_py_err)
}

fn load_calib(model_dir: &str) -> Option<Calibration> {
    // calib.json 的 per_class / per_lang 是对象，直接反序列化成
    // `Calibration { per_class: Option<Vec<f32>> }` 会整体失败、连温度一起丢掉。
    // Rust 后处理目前只用 temperature，这里单独取它，对其余字段保持宽容。
    let s = std::fs::read_to_string(std::path::Path::new(model_dir).join("calib.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&s).ok()?;
    Some(Calibration {
        temperature: v.get("temperature").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32,
        per_class: None,
    })
}

/// Rust 推理引擎（进程内唯一实例的句柄）。
#[pyclass]
pub struct EmotionEngine {
    inner: Arc<Engine>,
    calib: Option<Calibration>,
    precision: String,
}

#[pymethods]
impl EmotionEngine {
    #[new]
    #[pyo3(signature = (
      model_dir,
      tokenizer_dir=None,
      num_threads=4,
      max_batch=32,
      max_wait_ms=2,
      backend="onnx",
      precision="fp32",
      max_seq_len=64,
      use_lang_token=false,
      model_path=None,
      model=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model_dir: &str,
        tokenizer_dir: Option<&str>,
        num_threads: usize,
        max_batch: usize,
        max_wait_ms: u64,
        backend: &str,
        precision: &str,
        max_seq_len: usize,
        use_lang_token: bool,
        model_path: Option<String>,
        model: Option<&[u8]>,
    ) -> PyResult<Self> {
        let cfg = build_config(
            model_dir,
            tokenizer_dir,
            num_threads,
            max_batch,
            max_wait_ms,
            backend,
            precision,
            max_seq_len,
            use_lang_token,
            model_path,
            model,
        )?;
        let inner = load_global(cfg)?;
        let calib = load_calib(model_dir);
        Ok(Self {
            inner,
            calib,
            precision: precision.to_string(),
        })
    }

    /// 传入已加载的 ONNX 字节（模型由程序管理；仍受单一实例约束）。
    #[staticmethod]
    #[pyo3(signature = (model, model_dir, num_threads=4, max_batch=32, max_wait_ms=2,
                        precision="fp32", max_seq_len=64, use_lang_token=false, tokenizer_dir=None))]
    fn from_bytes(
        model: &[u8],
        model_dir: &str,
        num_threads: usize,
        max_batch: usize,
        max_wait_ms: u64,
        precision: &str,
        max_seq_len: usize,
        use_lang_token: bool,
        tokenizer_dir: Option<&str>,
    ) -> PyResult<Self> {
        let cfg = build_config(
            model_dir,
            tokenizer_dir,
            num_threads,
            max_batch,
            max_wait_ms,
            "onnx",
            precision,
            max_seq_len,
            use_lang_token,
            None,
            Some(model),
        )?;
        let inner = load_global(cfg)?;
        let calib = load_calib(model_dir);
        Ok(Self {
            inner,
            calib,
            precision: precision.to_string(),
        })
    }

    fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }

    fn num_labels(&self) -> usize {
        self.inner.num_labels()
    }

    fn precision(&self) -> &str {
        &self.precision
    }

    /// 文本 -> logits（分词 + 动态 padding + 前向 全在内核完成）。释放 GIL。
    #[pyo3(signature = (texts, lang=None))]
    fn forward<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
        lang: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let langs = resolve_langs(&texts, lang)?;
        let out = py
            .allow_threads(|| self.inner.logits(&texts, &langs))
            .map_err(to_py_err)?;
        let n = out.len();
        let k = out.first().map(|r| r.len()).unwrap_or(0);
        let flat: Vec<f32> = out.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((n, k), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }

    /// 一步到位：分词 -> 前向 -> softmax(含温度) -> top-k。释放 GIL。
    /// 与 Python 侧 `EmotionClassifier.predict` 语义一致：始终返回 k 个结果。
    #[pyo3(signature = (texts, top_k=3, lang=None))]
    fn predict(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        top_k: usize,
        lang: Option<Vec<String>>,
    ) -> PyResult<Vec<Vec<(u16, f32)>>> {
        let langs = resolve_langs(&texts, lang)?;
        let logits = py
            .allow_threads(|| self.inner.logits(&texts, &langs))
            .map_err(to_py_err)?;
        Ok(logits
            .iter()
            .map(|row| {
                let p = probs(row, self.calib.as_ref());
                top_k_scores(&p, top_k)
            })
            .collect())
    }

    /// 零拷贝高级接口：numpy 的 ids/mask 直入，logits 直出。
    fn forward_ids<'py>(
        &self,
        py: Python<'py>,
        ids: PyReadonlyArray2<'py, i64>,
        mask: PyReadonlyArray2<'py, i64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let ids_a = ids.as_array();
        let mask_a = mask.as_array();
        if ids_a.shape() != mask_a.shape() {
            return Err(PyValueError::new_err("ids 与 mask 形状不一致"));
        }
        let b = ids_a.nrows();
        let n = ids_a.ncols();
        let mut v_ids: Vec<Vec<u32>> = Vec::with_capacity(b);
        let mut v_mask: Vec<Vec<u32>> = Vec::with_capacity(b);
        for r in 0..b {
            let mut ri = Vec::with_capacity(n);
            let mut rm = Vec::with_capacity(n);
            for c in 0..n {
                let id = ids_a[[r, c]];
                let m = mask_a[[r, c]];
                if id < 0 || m < 0 {
                    return Err(PyValueError::new_err("ids/mask 不得为负数"));
                }
                ri.push(id as u32);
                rm.push(m as u32);
            }
            v_ids.push(ri);
            v_mask.push(rm);
        }
        let out = py
            .allow_threads(|| self.inner.logits_from_ids(&v_ids, &v_mask))
            .map_err(to_py_err)?;
        let k = out.first().map(|r| r.len()).unwrap_or(0);
        let flat: Vec<f32> = out.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((b, k), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }

    /// 仅解除本句柄引用；真正释放请调用模块级 `release_engine()`。
    fn close(&self) {}

    fn __repr__(&self) -> String {
        format!(
            "<EmotionEngine labels={} hidden={} precision={} global=true>",
            self.inner.num_labels(),
            self.inner.hidden_size(),
            self.precision
        )
    }
}

fn resolve_langs(texts: &[String], lang: Option<Vec<String>>) -> PyResult<Vec<String>> {
    match lang {
        Some(v) => {
            if v.len() != texts.len() {
                return Err(PyValueError::new_err("lang 数量与 texts 不一致"));
            }
            v.iter()
                .map(|l| {
                    normalize_lang(l)
                        .map(|s| s.to_string())
                        .ok_or_else(|| PyValueError::new_err(format!("不支持的语言: {l}")))
                })
                .collect()
        }
        None => Ok(texts.iter().map(|t| detect_lang(t).to_string()).collect()),
    }
}

/// 单条文本的语言检测（供 Python 侧核对）。
#[pyfunction]
fn detect(text: &str) -> String {
    detect_lang(text).to_string()
}

/// 释放进程内唯一引擎（换档 / 热更新前必须调用）。
#[pyfunction]
fn release_engine() -> bool {
    release_global_engine()
}

/// 查询是否已有唯一实例。
#[pyfunction]
fn has_engine() -> bool {
    emotion_runtime::has_global_engine()
}

#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EmotionEngine>()?;
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    m.add_function(wrap_pyfunction!(release_engine, m)?)?;
    m.add_function(wrap_pyfunction!(has_engine, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "MODEL_FILES",
        emotion_core::MODEL_FILES
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect::<std::collections::HashMap<&str, &str>>(),
    )?;
    Ok(())
}
