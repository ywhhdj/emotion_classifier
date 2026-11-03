"""回译增强（Back-Translation）。

zh -> en -> zh
ja -> en -> ja
en -> de/fr -> en

回译产出的是「语义不变、句法多样」的样本，不会像 EDA 那样破坏中日文语法。
结果落盘到 data/bt_cache.jsonl，只跑一次；CI 只读缓存。

质量闸门（任一不过即丢弃，不做兜底）：
  1) 往返 char-3gram Jaccard ∈ [0.45, 0.95]（过近=无增益，过远=语义漂移）
  2) 长度比 ∈ [0.6, 1.7]
  3) 语言检测与源语言一致
  4) 去重
"""

import json
import os
import random
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .config import Config

# NLLB-200 语言代码
NLLB_CODE = {
    "zh": "zho_Hans",
    "en": "eng_Latn",
    "ja": "jpn_Japa",
    "de": "deu_Latn",
    "fr": "fra_Latn",
}

# 每种语言的 pivot 侧语言（en 走 de/fr，避免 en->en 空转）
PIVOT_SIDE = {
    "zh": ["eng_Latn"],
    "ja": ["eng_Latn"],
    "en": ["deu_Latn", "fra_Latn"],
}


def _char_ngrams(text: str, n: int = 3) -> set:
    t = "".join(ch for ch in text if not ch.isspace())
    if len(t) <= n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def _jaccard(a: str, b: str, n: int = 3) -> float:
    sa, sb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def detect_lang(text: str) -> str:
    """轻量规则：假名 -> ja；汉字 -> zh；否则 en。"""
    kana = sum(1 for c in text if "\u3040" <= c <= "\u30ff")
    han = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if kana > 0:
        return "ja"
    if han > 0:
        return "zh"
    return "en"


class BackTranslator:
    """NLLB 回译器（离线、可复现）。

    依赖：transformers + sentencepiece。首次运行会加载
    facebook/nllb-200-distilled-600M（约 2.4GB fp32 / 1.2GB fp16）。
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        cache_path: Optional[str] = None,
        fp16: bool = False
    ):
        self.model_name = model_name or Config.BT_MODEL
        self.device = device
        self.cache_path = cache_path or Config.BT_CACHE_PATH
        self.fp16 = fp16 and device == "cuda"
        self._tok = None
        self._model = None
        self._cache: Dict[str, List[str]] = {}
        self._load_cache()

    # ── 缓存 ──
    def _load_cache(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("candidates"):
                    self._cache[rec["src"]] = rec["candidates"]

    def _append_cache(self, src: str, lang: str, label: str, candidates: List[str]):
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "src": src,
                "lang": lang,
                "label": label,
                "candidates": candidates
            }, ensure_ascii=False) + "\n")

    # ── 模型 ──
    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("回译需要 transformers：pip install transformers sentencepiece") from e
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        if self.fp16:
            self._model = self._model.half()
        self._model.to(self.device).eval()

    @staticmethod
    def _forced_bos(tok, code: str):
        """兼容不同 transformers 版本的 forced_bos_token_id 取法。"""
        try:
            return tok.convert_tokens_to_ids(code)
        except Exception:
            return None

    def translate(self, texts: Sequence[str], src: str, tgt: str,
                  num_return: int = 2, max_new_tokens: int = 96) -> List[str]:
        import torch
        self._ensure_model()
        src_code, tgt_code = NLLB_CODE[src], NLLB_CODE[tgt] if tgt in NLLB_CODE else tgt
        batch = self._tok(list(texts), return_tensors="pt", padding=True,
                          truncation=True, max_length=128).to(self.device)
        bos = self._forced_bos(self._tok, tgt_code)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": max(2, num_return * 2),
            "num_return_sequences": num_return,
            "no_repeat_ngram_size": 3,
            "do_sample": num_return > 1
        }
        if bos is not None:
            gen_kwargs["forced_bos_token_id"] = bos
        self._model.generation_config.max_length = None
        with torch.no_grad():
            out = self._model.generate(**batch, **gen_kwargs)
        return [self._tok.decode(g, skip_special_tokens=True) for g in out]

    # ── 回译 ──
    def round_trip(self, text: str, lang: str, factor: int = 1) -> List[str]:
        """一次往返：lang -> pivot -> lang，返回通过质量闸门的候选。"""
        cached = self._cache.get(text)
        if cached is not None:
            return cached[:max(1, factor)]

        pivots = PIVOT_SIDE.get(lang, ["eng_Latn"])
        pivot_lang = {"eng_Latn": "en", "deu_Latn": "de", "fra_Latn": "fr"}.get(pivots[0], "en")

        try:
            mid = self.translate([text], src=lang, tgt=pivot_lang, num_return=1)
            if not mid:
                return []
            back = self.translate(mid * 2, src=pivot_lang, tgt=lang, num_return=2)
        except Exception as e:
            print(f"[bt] 翻译失败（{e}），跳过该样本")
            return []

        kept = []
        for cand in back:
            cand = (cand or "").strip()
            if not cand or cand == text:
                continue
            if detect_lang(cand) != lang:
                continue
            j = _jaccard(text, cand)
            if not (Config.BT_MIN_JACCARD <= j <= Config.BT_MAX_JACCARD):
                continue
            ratio = len(cand) / max(1, len(text))
            if not (Config.BT_MIN_LEN_RATIO <= ratio <= Config.BT_MAX_LEN_RATIO):
                continue
            if cand not in kept:
                kept.append(cand)
        return kept[:max(1, factor)]

    def augment(self, texts: Sequence[str], langs: Sequence[str], labels: Sequence[str],
                factor: int = 1) -> List[dict]:
        out = []
        for t, lg, lb in zip(texts, langs, labels):
            lg = (lg or detect_lang(t)).lower()
            if lg not in ("zh", "en", "ja"):
                continue
            cands = self.round_trip(t, lg, factor=factor)
            for c in cands:
                out.append({
                    "text": c,
                    "label": lb,
                    "lang": lg, 
                    "is_mt": 1
                })
            if cands:
                self._append_cache(t, lg, lb, cands)
        return out


def augment_with_backtranslation(
    df: pd.DataFrame,
    factor: int = 1,
    model_name: Optional[str] = None,
    cache_path: Optional[str] = None,
    device: str = "cpu"
) -> pd.DataFrame:
    """对训练集做回译增强，返回拼接后的 DataFrame（只影响传入的 df）。"""
    bt = BackTranslator(model_name=model_name, device=device, cache_path=cache_path)
    texts = df["text"].astype(str).tolist()
    langs = df["lang"].astype(str).tolist() if "lang" in df.columns else [None] * len(df)
    labels = df["label"].astype(str).tolist()
    rows = bt.augment(texts, langs, labels, factor=factor)
    if not rows:
        print("[bt] 无回译样本产出（检查 NLLB 是否已下载 / 闸门是否过严）")
        return df
    print(f"[bt] 回译产出 {len(rows)} 条（原 {len(df)}）")
    new = pd.DataFrame(rows)
    new["label_id"] = new["label"].map(df.drop_duplicates("label").set_index("label")["label_id"])
    merged = pd.concat([df, new.dropna(subset=["label_id"])], ignore_index=True)
    merged = merged.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return merged
