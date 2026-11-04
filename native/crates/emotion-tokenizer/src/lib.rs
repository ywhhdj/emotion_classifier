//! 分词与预处理（v0.2.0）。
//!
//! 归一化顺序与 `train/train.py::TextCleaner.clean` 对齐：
//!   NFKC -> 控制/零宽/全角空格清理 -> HTML/URL/装饰符清理 -> 空白折叠
//!   -> lang 判定 -> 拼 <lang> 前缀 -> spm encode -> 截断到 max_seq_len
//!
//! **不再把每条固定 pad 到 max_seq_len**：由 `Engine::pad_batch` 按 batch 内
//! 最长动态 padding（与训练侧 `make_collate` 一致），极短文本可省 3~5 倍算力。

use anyhow::{Context, Result};
use emotion_core::{detect_lang, lang_token};
use rayon::prelude::*;
use sentencepiece_rs::SentencePieceProcessor;
use std::collections::HashMap;
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
    lang_ids: HashMap<&'static str, u32>,
}

pub struct Encoded {
    pub ids: Vec<u32>,
    pub mask: Vec<u32>,
}

impl Tokenizer {
    pub fn from_dir(
        model_dir: &Path,
        max_seq_len: usize,
        use_lang_token: bool
    ) -> Result<Self> {
        let path = Self::find_model(model_dir)
            .with_context(|| format!("未找到 sentencepiece 模型: {}", model_dir.display()))?;
        let spp = SentencePieceProcessor::open(&path)
            .with_context(|| format!("加载 sentencepiece 失败: {}", path.display()))?;

        let to_u32 = |id: usize| -> u32 { id as u32 };

        let bos_id = spp.bos_id().map(to_u32).unwrap_or(0);
        let eos_id = spp.eos_id().map(to_u32).unwrap_or(2);
        let pad_id = spp.pad_id().map(to_u32).unwrap_or(1);
        let unk_id = to_u32(spp.unk_id());

        // piece_to_id 返回 usize，需要转换并检查有效性
        let mask_id = {
            let raw = to_u32(spp.piece_to_id("<mask>"));
            if raw == unk_id {
                250_001 // 与训练侧一致的回退值
            } else {
                raw
            }
        };

        let mut lang_ids = HashMap::new();
        if use_lang_token {
            for (l, t) in emotion_core::LANG_TOKENS {
                let id = spp.piece_to_id(t);
                if id < spp.encode("").unwrap().len() {
                    lang_ids.insert(l, id as u32);
                }
            }
        }

        Ok(Self {
            spp, bos_id, eos_id, pad_id, unk_id, mask_id,
            max_seq_len, use_lang_token, lang_ids,
        })
    }

    fn find_model(model_dir: &Path) -> Option<PathBuf> {
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
        self.spp.encode("").unwrap().len()
    }

    fn strip_html(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut in_tag = false;
        for c in s.chars() {
            match c {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => out.push(c),
                _ => {}
            }
        }
        out
    }

    fn strip_urls(s: &str) -> String {
        s.split_whitespace()
            .filter(|w| {
                !(w.starts_with("http://") || w.starts_with("https://")
                  || w.starts_with("www."))
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// 与训练侧 TextCleaner.clean 对齐的归一化（不引入 regex 依赖）。
    pub fn normalize(text: &str) -> String {
        let s: String = text.nfkc().collect();
        let s: String = s
            .chars()
            .filter(|c| !c.is_control())
            .filter(|c| !matches!(c,
                '\u{200b}'..='\u{200d}' | '\u{2028}'..='\u{202f}' |
                '\u{feff}' | '\u{3000}'))
            .collect();
        let s = Self::strip_html(&s);
        let s = Self::strip_urls(&s);
        let s: String = s
            .chars()
            .filter(|c| !matches!(c, '【' | '】' | '■' | '◆' | '▶' | '☞' | '➤'))
            .collect();
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    pub fn encode_one(&self, text: &str, lang: Option<&str>) -> Encoded {
        let norm = Self::normalize(text);
        let lang = lang
            .map(|l| l.to_string())
            .unwrap_or_else(|| detect_lang(&norm).to_string());

        let lang_id = if self.use_lang_token {
            self.lang_ids.get(lang.as_str()).copied()
        } else {
            None
        };

        let reserved = 2 + usize::from(lang_id.is_some());
        let limit = self.max_seq_len.saturating_sub(reserved);
        let mut ids: Vec<u32> = Vec::with_capacity(self.max_seq_len);
        if let Some(id) = lang_id {
            ids.push(id);
        }
        ids.push(self.bos_id);

        // 改用 encode_to_ids，直接得到 Vec<u32>
        let pieces = self.spp.encode_to_ids(&norm).unwrap_or_default();
        for id in pieces.into_iter().take(limit) {
            ids.push(id as u32);
        }
        ids.push(self.eos_id);

        let mask = vec![1u32; ids.len()];
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

/// 供外部核对语言标记（与 emotion_core::lang_token 保持一致）。
pub fn token_for_lang(lang: &str) -> Option<&'static str> {
    lang_token(lang)
}
