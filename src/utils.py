import numpy as np
import torch

def mixup(
    x,
    y,
    alpha=0.4
):
    """MixUp 混合训练。"""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[index]
    return mixed, y, y[index], lam