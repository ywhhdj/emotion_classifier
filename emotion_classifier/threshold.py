"""概率校准：temperature scaling + 可选 per-class / per-language 阈值。

calib.json 结构：
{
  "temperature": 1.23,
  "per_class": {"害怕": 0.42, ...},
  "per_lang": {"ja": {"temperature": 1.4}, ...}
}

内核（Rust）已内置 temperature 与阈值；本模块供 Python 回退路径
（ONNX/PyTorch）与离线分析使用。
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class Calibration:
    def __init__(self, temperature: float = 1.0,
                 per_class: Optional[Dict[str, float]] = None,
                 per_lang: Optional[Dict[str, dict]] = None):
        self.temperature = float(temperature)
        self.per_class = per_class or {}
        self.per_lang = per_lang or {}

    @classmethod
    def load(cls, path) -> "Calibration":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            temperature=float(d.get("temperature", 1.0)),
            per_class=d.get("per_class") or {},
            per_lang=d.get("per_lang") or {}
        )

    def temperature_for(self, lang: Optional[str] = None) -> float:
        if lang and lang in self.per_lang:
            return float(self.per_lang[lang].get("temperature", self.temperature))
        return self.temperature

    def apply(self, logits: np.ndarray, lang: Optional[str] = None) -> np.ndarray:
        """对 logits 施加温度缩放（不改变 argmax，但影响置信度与阈值判定）。"""
        T = self.temperature_for(lang)
        if abs(T - 1.0) < 1e-6:
            return logits
        return logits / T

    def thresholds(self, id2label: Dict[int, str], lang: Optional[str] = None) -> Optional[np.ndarray]:
        """返回按 label_id 排列的 per-class 阈值数组（无配置时返回 None）。"""
        if not self.per_class:
            return None
        n = len(id2label)
        out = np.full(n, 0.5, dtype=np.float32)
        for i in range(n):
            name = id2label.get(i)
            if name in self.per_class:
                out[i] = float(self.per_class[name])
        return out


def find_calib(model_dir) -> Optional[Path]:
    p = Path(model_dir) / "calib.json"
    return p if p.exists() else None


def load_calib(model_dir, env_path: Optional[str] = None) -> Calibration:
    """优先环境变量，其次模型目录；都没有则退化为 T=1.0。"""
    if env_path:
        return Calibration.load(env_path)
    p = find_calib(model_dir)
    return Calibration.load(p) if p else Calibration()
