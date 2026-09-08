import random
from typing import Literal
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def mixup(x, y, alpha=0.4):
    """MixUp 混合训练。"""
    if alpha is None or alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    # 保证主导样本占比不低于 0.5，避免小数据集上过度混合
    lam = max(lam, 1.0 - lam)
    index = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[index]
    return mixed, y, y[index], lam


def class_weights(
    counts: np.ndarray,
    strategy: Literal["inverse", "sqrt", "effective"] = "effective",
    beta: float = 0.999
) -> np.ndarray:
    """类别权重策略。

    inverse   : 1/n        —— 对大类惩罚过重
    sqrt      : 1/sqrt(n)  —— 更温和
    effective : (1-beta)/(1-beta^n)（Cui et al. 2019）—— 默认
    """
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.clip(counts, 1.0, None)
    if strategy == "none":
        w = np.ones_like(counts)
    elif strategy == "inverse":
        w = 1.0 / counts
    elif strategy == "sqrt":
        w = 1.0 / np.sqrt(counts)
    elif strategy == "effective":
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        raise ValueError(f"未知类别权重策略: {strategy}")
    return w / w.sum() * len(counts)


def macro_f1(y_true, y_pred, num_labels: int) -> float:
    """不依赖 sklearn 的 macro-F1，便于在任意环境复用。"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    f1s = []
    for c in range(num_labels):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))