//! 内核公共类型：配置、三档模型、标签、语言、错误（v0.2.0）。

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const VERSION: &str = "0.2.0";
pub const SUPPORTED_LANGS: [&str; 3] = ["zh", "en", "ja"];
pub const LANG_TOKENS: [(&str, &str); 3] = [("zh", "<zh>"), ("en", "<en>"), ("ja", "<ja>")];

/// 与 Release asset 名严格对应的三档 ONNX 整图。
pub const MODEL_FILES: [(&str, &str); 3] = [
    ("fp32", "emotion_classifier.onnx"),
    ("fp16", "emotion_classifier_fp16.onnx"),
    ("int8", "emotion_classifier_int8.onnx"),
];

pub fn model_filename(precision: Precision) -> &'static str {
    match precision {
        Precision::Fp32 => MODEL_FILES[0].1,
        Precision::Fp16 => MODEL_FILES[1].1,
        Precision::Int8 => MODEL_FILES[2].1,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("模型目录无效或缺失文件: {0}")]
    ModelLoad(String),
    #[error("分词失败: {0}")]
    Tokenizer(String),
    #[error("前向失败: {0}")]
    Forward(String),
    #[error("输入非法: {0}")]
    InvalidInput(String),
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON 错误: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendKind {
    /// 纯 Rust 前向（需 model.safetensors；Release 默认不提供，属可选档）
    Candle,
    /// onnxruntime 加载三档 ONNX 整图（v0.2.0 默认）
    Onnx,
}

impl Default for BackendKind {
    fn default() -> Self {
        BackendKind::Onnx
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Precision {
    Fp32,
    Fp16,
    Int8,
}

impl Default for Precision {
    fn default() -> Self {
        Precision::Fp32
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub model_dir: PathBuf,
    pub num_threads: usize,
    pub tokenizer_dir: Option<PathBuf>,
    pub max_batch: usize,
    pub max_wait_ms: u64,
    pub backend: BackendKind,
    pub precision: Precision,
    /// 与 train/config.py 对齐：默认 64 / 不加 lang token
    pub max_seq_len: usize,
    pub use_lang_token: bool,
    /// 调用方显式指定的 ONNX 路径（覆盖 precision -> 文件名的映射）
    pub model_path: Option<PathBuf>,
    /// 调用方传入的「已加载模型」字节（由程序管理，避免重复读盘）
    pub model_bytes: Option<Vec<u8>>,
    /// 外部模型的指纹（参与单例配置校验，防止误复用）
    pub fingerprint: Option<String>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("models"),
            tokenizer_dir: None,
            num_threads: 4,
            max_batch: 32,
            max_wait_ms: 2,
            backend: BackendKind::default(),
            precision: Precision::default(),
            max_seq_len: 64,
            use_lang_token: false,
            model_path: None,
            model_bytes: None,
            fingerprint: None,
        }
    }
}

impl EngineConfig {
    /// 进程内单一实例的身份指纹：配置不同即拒绝复用（不重复初始化模型）。
    pub fn key(&self) -> String {
        format!(
            "{}|{:?}|{:?}|{}|{}|{}|{}|{}|{}|{}",
            self.model_dir.display(),
            self.backend,
            self.precision,
            self.num_threads,
            self.max_batch,
            self.max_wait_ms,
            self.max_seq_len,
            self.use_lang_token,
            self.model_path.as_ref().map(|p| p.display().to_string()).unwrap_or_default(),
            self.fingerprint.as_deref().unwrap_or(""),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub temperature: f32,
    pub per_class: Option<Vec<f32>>,
}

impl Default for Calibration {
    fn default() -> Self {
        Self { temperature: 1.0, per_class: None }
    }
}

impl Calibration {
    pub fn temperature(&self) -> f32 {
        if self.temperature <= 0.0 { 1.0 } else { self.temperature }
    }
}

/// 单条预测结果：`(label_id, score)`。
pub type Scored = (u16, f32);

/// 轻量规则语言检测：假名 -> ja；汉字 -> zh；否则 en。
pub fn detect_lang(text: &str) -> &'static str {
    let kana = text
        .chars()
        .filter(|c| ('\u{3040}'..='\u{30ff}').contains(c))
        .count();
    if kana > 0 {
        return "ja";
    }
    let han = text
        .chars()
        .filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c))
        .count();
    if han > 0 {
        return "zh";
    }
    "en"
}

pub fn lang_token(lang: &str) -> Option<&'static str> {
    LANG_TOKENS.iter().find(|(l, _)| *l == lang).map(|(_, t)| *t)
}

pub fn normalize_lang(lang: &str) -> Option<&'static str> {
    match lang.trim().to_lowercase().as_str() {
        "zh" | "zh-cn" | "zh_cn" | "chinese" | "cn" | "中文" => Some("zh"),
        "en" | "en-us" | "eng" | "english" | "英文" => Some("en"),
        "ja" | "jp" | "ja-jp" | "japanese" | "日文" | "日语" => Some("ja"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_language() {
        assert_eq!(detect_lang("怖くて震えが止まらない"), "ja");
        assert_eq!(detect_lang("脸颊泛红"), "zh");
        assert_eq!(detect_lang("i'm so excited"), "en");
    }

    #[test]
    fn three_model_files() {
        assert_eq!(model_filename(Precision::Fp32), "emotion_classifier.onnx");
        assert_eq!(model_filename(Precision::Fp16), "emotion_classifier_fp16.onnx");
        assert_eq!(model_filename(Precision::Int8), "emotion_classifier_int8.onnx");
    }

    #[test]
    fn config_key_changes_with_precision() {
        let a = EngineConfig::default();
        let mut b = a.clone();
        b.precision = Precision::Int8;
        assert_ne!(a.key(), b.key());
    }
}
