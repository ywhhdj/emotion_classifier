import torch
from torch import nn

class StochasticDepth(nn.Module):
    """随机深度：训练时以概率 p 丢弃整个残差分支。"""

    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep_prob = 1 - self.p
        mask = torch.empty(x.size(0), 1, device=x.device).bernoulli_(keep_prob)
        return x / keep_prob * mask.to(x.dtype)


class EmotionClassifier(nn.Module):
    """推理侧分类头，结构需与 train/models.py 的 LightEmotionClassifier 保持一致。"""

    def __init__(self, input_dim=384, num_classes=19, hidden_dim=128,
                 dropout=0.3, stochastic_p=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stochastic_depth = StochasticDepth(p=stochastic_p)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, label=None):
        x = self.proj(x)
        residual = self.stochastic_depth(self.residual(x))
        x = x + residual
        logits = self.classifier(x)
        return logits, x, None