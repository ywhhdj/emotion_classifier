"""跨语言扩充（Cross-lingual transfer augmentation）。

数据现状（data/emotion.csv，2556 行）：
    zh 1863（72.9%） / en 356（13.9%） / ja 337（13.2%）
    en、ja 每类仅 16~36 条 —— 任何模型都学不出判别边界。

做法：把高资源语言（zh）的标注样本机器翻译到低资源语言（ja/en），
直接引入「新的语义样本」而不是同义改写，比语内增强更有效。

两条硬约束：
  1) MT 样本不超过真实样本的 CL_MAX_RATIO 倍（默认 1.0），只扩充不替换；
  2) 同一原文的多语译文共享 group_id，划分时整组划分，避免 train/test 泄漏。
"""

import hashlib
import os
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .config import Config
from .backtranslate import BackTranslator, NLLB_CODE, detect_lang


def _group_id(text: str) -> str:
    return "g_" + hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


class CrossLingualExpander:
    """zh/en -> ja/en 的翻译扩充器（复用 BackTranslator 的 NLLB 引擎）。"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        cache_path: Optional[str] = None
    ):
        self.bt = BackTranslator(model_name=model_name, device=device, cache_path=cache_path)

    def transfer(
        self,
        texts: Sequence[str],
        src_lang: str,
        tgt_lang: str,
        labels: Optional[Sequence[str]] = None
    ) -> List[dict]:
        if not texts:
            return []
        translated = self.bt.translate(
            list(texts),
            src=src_lang,
            tgt=tgt_lang,
            num_return=1, 
            max_new_tokens=96
        )
        rows = []
        for i, t in enumerate(translated):
            t = (t or "").strip()
            if not t:
                continue
            if detect_lang(t) != tgt_lang:      # 语言一致性闸门
                continue
            src = texts[i % len(texts)]
            rows.append({
                "text": t,
                "label": labels[i] if labels is not None else None,
                "lang": tgt_lang,
                "is_mt": 1,
                "group_id": _group_id(src),      # 与源样本同组，划分时整组走
            })
        return rows


def _quota_need(counts: Dict[str, int]) -> Dict[str, int]:
    """按 CL_TARGET_QUOTA 计算每种语言还缺多少条。"""
    need = {}
    for lang, mult in Config.CL_TARGET_QUOTA.items():
        cur = counts.get(lang, 0)
        need[lang] = max(0, int(cur * mult) - cur)
    return need


def expand_crosslingual(
    df: pd.DataFrame,
    device: str = "cpu",
    model_name: Optional[str] = None,
    cache_path: Optional[str] = None,
    max_ratio: Optional[float] = None
) -> pd.DataFrame:
    """按配额把高资源语言的样本翻译到低资源语言。

    返回：原 df（group_id/is_mt 补全）+ 新增 MT 行。
    """
    max_ratio = Config.CL_MAX_RATIO if max_ratio is None else max_ratio
    if "lang" not in df.columns:
        print("[cl] 数据无 lang 列，跳过跨语言扩充")
        return df

    df = df.copy()
    df["lang"] = df["lang"].astype(str).str.lower()
    if "group_id" not in df.columns:
        df["group_id"] = df["text"].map(_group_id)
    if "is_mt" not in df.columns:
        df["is_mt"] = 0

    counts = df["lang"].value_counts().to_dict()
    need = _quota_need(counts)
    if all(v == 0 for v in need.values()):
        print("[cl] 各语言配额已满足，跳过")
        return df

    expander = CrossLingualExpander(model_name=model_name, device=device, cache_path=cache_path)
    new_rows: List[dict] = []

    for src_lang, tgt_lang in Config.CL_TRANSFER:
        if need.get(tgt_lang, 0) <= 0:
            continue
        pool = df[(df["lang"] == src_lang) & (df["is_mt"] == 0)]
        if pool.empty:
            continue
        take = min(need[tgt_lang], int(len(pool) * max_ratio), len(pool))
        if take <= 0:
            continue
        sample = pool.sample(n=take, random_state=Config.RANDOM_STATE)
        rows = expander.transfer(
            sample["text"].astype(str).tolist(),
            src_lang,
            tgt_lang,
            sample["label"].astype(str).tolist()
        )
        # 标签 id 回填
        l2i = df.drop_duplicates("label").set_index("label")["label_id"].to_dict()
        for r in rows:
            r["label_id"] = l2i.get(r["label"])
        rows = [r for r in rows if r.get("label_id") is not None]
        new_rows.extend(rows)
        need[tgt_lang] -= len(rows)
        print(f"[cl] {src_lang} -> {tgt_lang}: +{len(rows)}")

    if not new_rows:
        return df

    out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    out = out.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    print(f"[cl] 跨语言扩充完成: {len(df)} → {len(out)}")
    print("[cl] 语言分布: " + str(out["lang"].value_counts().to_dict()))
    return out
