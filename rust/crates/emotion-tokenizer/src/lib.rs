//! 分词与预处理。
//!
//! 归一化顺序**必须与训练期 Python 侧逐字对齐**（CI 有 1 万条一致性测试）：
//!   NFKC -> 空白折叠 -> 控制字符清理 -> 保留 emoji -> lang 判定
//!   -> 拼 <lang> 前缀 -> spm encode -> 截断到 max_seq_len
//!
//! 分词器直接加载裁剪产物自带的 `sentencepiece.bpe.model`，
//! 与训练期完全一致，避免 HF / Rust 两套分词漂移。

use anyhow::{Context, Result};
use emotion_core::{detect_lang, lang_token};
use rayon::prelude::*;
use sentencepiece::SentencePieceProcessor;
use std::path::{Path, PathBuf};
use unicode_normalization::UnicodeNormalization;

pub struct Tokenizer {
    spp: SentencePieceProcessor,
    pub bos_id: u32,
    pub eos_id: u32,
    pub pad_id: u32,
    pub unk_id: u32,
    pub mask_id: u32,
    pub max_seq_len: usize,
    pub use_lang_token: bool,
    lang_ids: std::collections::HashMap<&'static str, u32>,
}

pub struct Encoded {
    pub ids: Vec<u32>,
    pub mask: Vec<u32>,
}

impl Tokenizer {
    pub fn from_dir(model_dir: &Path, max_seq_len: usize, use_lang_token: bool) -> Result<Self> {
        let path = Self::find_model(model_dir)
            .with_context(|| format!("未找到 sentencepiece 模型: {}", model_dir.display()))?;
        let spp = SentencePieceProcessor::open(&path)
            .with_context(|| format!("加载 sentencepiece 失败: {}", path.display()))?;

        let pi = |id: Option<u32>, default: u32| id.unwrap_or(default);
        let bos_id = pi(spp.bos_id(), 0);
        let eos_id = pi(spp.eos_id(), 2);
        let pad_id = pi(spp.pad_id(), 1);
        let unk_id = pi(spp.unk_id(), 3);
        let mask_id = pi(spp.unk_id(), 250001);

        let mut lang_ids = std::collections::HashMap::new();
        if use_lang_token {
            for (l, t) in emotion_core::LANG_TOKENS {
                if let Some(id) = spp.piece_to_id(t) {
                    lang_ids.insert(l, *id);
                }
            }
        }

        Ok(Self {
            spp,
            bos_id,
            eos_id,
            pad_id,
            unk_id,
            mask_id,
            max_seq_len,
            use_lang_token,
            lang_ids,
        })
    }

    fn find_model(model_dir: &Path) -> Option<PathBuf> {
        // 训练产物自带自洽的 .model（三语裁剪后重新导出）
        let candidates = [
            model_dir.join("sentencepiece.bpe.model"),
            model_dir.join("tokenizer").join("sentencepiece.bpe.model"),
            model_dir.join("spm.model"),
            model_dir.join("source.spm"),
        ];
        candidates.into_iter().find(|p| p.exists()).or_else(|| {
            std::fs::read_dir(model_dir.join("tokenizer"))
                .ok()
                .and_then(|rd| {
                    rd.filter_map(Result::ok)
                        .map(|e| e.path())
                        .find(|p| p.extension().map(|e| e == "model").unwrap_or(false))
                })
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.spp.len() as usize
    }

    /// 与 Python 侧 `TextCleaner.clean` 对齐的归一化。
    pub fn normalize(text: &str) -> String {
        let s: String = text.nfkc().collect();
        let s: String = s
            .chars()
            .filter(|c| !c.is_control())
            .filter(|c| !matches!(c,
                '\u{200b}'..='\u{200d}' | '\u{2028}'..='\u{202f}' | '\u{feff}' | '\u{3000}'))
            .collect();
        // 空白折叠（保留 emoji：不做任何字符类过滤）
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    pub fn encode_one(&self, text: &str, lang: Option<&str>) -> Encoded {
        let norm = Self::normalize(text);
        let lang = lang.map(|l| l.to_string()).unwrap_or_else(|| detect_lang(&norm).to_string());

        let mut ids: Vec<u32> = Vec::with_capacity(self.max_seq_len);
        if self.use_lang_token {
            if let Some(id) = self.lang_ids.get(lang.as_str()) {
                ids.push(*id);
            }
        }
        ids.push(self.bos_id);
        let pieces = self.spp.encode(&norm).unwrap_or_default();
        let limit = self.max_seq_len.saturating_sub(2);
        for p in pieces.into_iter().take(limit) {
            ids.push(p.id);
        }
        ids.push(self.eos_id);

        let len = ids.len();
        let mut mask = vec![1u32; len];
        if len < self.max_seq_len {
            ids.resize(self.max_seq_len, self.pad_id);
            mask.resize(self.max_seq_len, 0);
        } else {
            ids.truncate(self.max_seq_len);
            mask.truncate(self.max_seq_len);
        }
        Encoded { ids, mask }
    }

    pub fn encode_batch(&self, texts: &[String], langs: &[String]) -> Vec<Encoded> {
        texts
            .par_iter()
            .zip(langs.par_iter())
            .map(|(t, l)| self.encode_one(t, Some(l)))
            .collect()
    }
}
