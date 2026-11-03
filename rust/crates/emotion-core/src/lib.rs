//! 内核公共类型：配置、标签、语言、错误。

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const SUPPORTED_LANGS: [&str; 3] = ["zh", "en", "ja"];
pub const LANG_TOKENS: [(&str, &str); 3] = [("zh", "<zh>"), ("en", "<en>"), ("ja", "<ja>")];

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
    /// 纯 Rust 推理（默认；易交叉编译，无 C 依赖）
    Candle,
    /// onnxruntime（feature backend-ort；服务端 GPU / 高吞吐）
    Onnx,
}

impl Default for BackendKind {
    fn default() -> Self {
        BackendKind::Candle
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
        Precision::Fp16
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub model_dir: PathBuf,
    pub num_threads: usize,
    pub max_batch: usize,
    pub max_wait_ms: u64,
    pub backend: BackendKind,
    pub precision: Precision,
    /// 最大序列长度（训练期 48；裁剪词表后仍保持一致）
    pub max_seq_len: usize,
    /// 是否在前缀拼接 <zh>/<en>/<ja>
    pub use_lang_token: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("models"),
            num_threads: 4,
            max_batch: 32,
            max_wait_ms: 2,
            backend: BackendKind::default(),
            precision: Precision::default(),
            max_seq_len: 48,
            use_lang_token: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Calibration {
    pub temperature: f32,
    pub per_class: Option<Vec<f32>>,
}

impl Calibration {
    pub fn temperature(&self) -> f32 {
        if self.temperature <= 0.0 {
            1.0
        } else {
            self.temperature
        }
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
}
