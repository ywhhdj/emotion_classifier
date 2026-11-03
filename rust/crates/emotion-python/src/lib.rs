//! PyO3 绑定 —— 编译为 `emotion_classifier._core`。
//!
//! 要点：
//!   * `py.allow_threads(...)` 释放 GIL，Python 多线程可真正并行调用；
//!   * numpy 零拷贝：`PyReadonlyArray2` 进、`PyArray2` 出；
//!   * abi3-py310：一份 wheel 覆盖多 Python 版本。

use emotion_core::{detect_lang, normalize_lang, BackendKind, Calibration, EngineConfig, Precision};
use emotion_postprocess::{probs, top_k_thresholded};
use emotion_runtime::Engine;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
        "fp32" => Ok(Precision::Fp32),
        "fp16" => Ok(Precision::Fp16),
        "int8" => Ok(Precision::Int8),
        other => Err(PyValueError::new_err(format!("未知精度: {other}"))),
    }
}

/// Rust 推理引擎。Python 侧只做参数校验与结果搬运。
#[pyclass]
pub struct EmotionEngine {
    inner: Engine,
    calib: Option<Calibration>,
}

#[pymethods]
impl EmotionEngine {
    #[new]
    #[pyo3(signature = (model_dir, num_threads=4, max_batch=32, max_wait_ms=2,
                        backend="candle", precision="fp16", max_seq_len=48, use_lang_token=true))]
    fn new(
        model_dir: &str,
        num_threads: usize,
        max_batch: usize,
        max_wait_ms: u64,
        backend: &str,
        precision: &str,
        max_seq_len: usize,
        use_lang_token: bool,
    ) -> PyResult<Self> {
        let cfg = EngineConfig {
            model_dir: std::path::PathBuf::from(model_dir),
            num_threads,
            max_batch,
            max_wait_ms,
            backend: parse_backend(backend)?,
            precision: parse_precision(precision)?,
            max_seq_len,
            use_lang_token,
        };
        let inner = Engine::load(cfg).map_err(to_py_err)?;

        // 校准文件（可选）
        let calib = std::fs::read_to_string(std::path::Path::new(model_dir).join("calib.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<Calibration>(&s).ok());

        Ok(Self { inner, calib })
    }

    fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }

    fn num_labels(&self) -> usize {
        self.inner.num_labels()
    }

    /// 文本 -> logits（分词 + 批调度 + 前向 全在内核完成）。释放 GIL。
    #[pyo3(signature = (texts, lang=None))]
    fn forward<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
        lang: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let langs = resolve_langs(&texts, lang)?;
        let out = py.allow_threads(|| self.inner.logits(&texts, &langs)).map_err(to_py_err)?;
        let n = out.len();
        let k = out.first().map(|r| r.len()).unwrap_or(0);
        let flat: Vec<f32> = out.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((n, k), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    /// 一步到位：分词 -> 前向 -> softmax/阈值 -> top-k。释放 GIL。
    #[pyo3(signature = (texts, top_k=3, lang=None))]
    fn predict(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
        top_k: usize,
        lang: Option<Vec<String>>,
    ) -> PyResult<Vec<Vec<(u16, f32)>>> {
        let langs = resolve_langs(&texts, lang)?;
        let logits = py.allow_threads(|| self.inner.logits(&texts, &langs)).map_err(to_py_err)?;
        let th = self.calib.as_ref().and_then(|c| c.per_class.as_deref());
        Ok(logits
            .iter()
            .map(|row| {
                let p = probs(row, self.calib.as_ref());
                top_k_thresholded(&p, top_k, th)
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
            v_ids.push((0..n).map(|c| ids_a[[r, c]] as u32).collect());
            v_mask.push((0..n).map(|c| mask_a[[r, c]] as u32).collect());
        }
        let out = py
            .allow_threads(|| {
                // 直接走 backend：此处借用 Engine 的内部批处理路径
                self.inner.logits_from_ids(&v_ids, &v_mask)
            })
            .map_err(to_py_err)?;
        let k = out.first().map(|r| r.len()).unwrap_or(0);
        let flat: Vec<f32> = out.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((b, k), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
    }

    fn close(&self) {
        self.inner.close();
    }

    fn __repr__(&self) -> String {
        format!(
            "<EmotionEngine labels={} hidden={}>",
            self.inner.num_labels(),
            self.inner.hidden_size()
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

#[pymodule]
fn _core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EmotionEngine>()?;
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    let _ = py;
    Ok(())
}
