"""语言归一化与轻量语言检测（纯规则，不引入额外模型）。

规则：假名 -> ja；汉字 -> zh；其余 -> en。覆盖 99% 场景。
"""

from typing import List, Optional

SUPPORTED_LANGS = ("zh", "en", "ja")

_ALIAS = {
    "zh": "zh", "zh-cn": "zh", "zh_cn": "zh", "chinese": "zh", "中文": "zh", "cn": "zh",
    "en": "en", "en-us": "en", "eng": "en", "english": "en", "英文": "en",
    "ja": "ja", "jp": "ja", "ja-jp": "ja", "japanese": "ja", "日文": "ja", "日语": "ja",
}

LANG_TOKENS = {"zh": "<zh>", "en": "<en>", "ja": "<ja>"}


def normalize_lang(lang: Optional[str]) -> Optional[str]:
    if lang is None:
        return None
    key = str(lang).strip().lower()
    return _ALIAS.get(key)


def normalize_langs(langs: Optional[List[str]], n: int) -> Optional[List[str]]:
    if langs is None:
        return None
    if isinstance(langs, str):
        langs = [langs]
    if len(langs) != n:
        raise ValueError(f"lang 数量({len(langs)}) 与文本数量({n}) 不一致")
    out = []
    for l in langs:
        nl = normalize_lang(l)
        if nl is None:
            raise ValueError(f"不支持的语言: {l}（支持 {SUPPORTED_LANGS}）")
        out.append(nl)
    return out


def detect_lang(text: str) -> str:
    """轻量规则检测。"""
    kana = sum(1 for c in text if "\u3040" <= c <= "\u30ff")
    han = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if kana > 0:
        return "ja"
    if han > 0:
        return "zh"
    return "en"


def detect_langs(texts: List[str]) -> List[str]:
    return [detect_lang(t or "") for t in texts]


def apply_lang_token(text: str, lang: Optional[str], enabled: bool = True) -> str:
    if not enabled or lang is None:
        return text
    tok = LANG_TOKENS.get(lang)
    return f"{tok}{text}" if tok else text
