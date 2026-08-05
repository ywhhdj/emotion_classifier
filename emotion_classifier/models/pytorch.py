import torch
from torch import nn

class StochasticDepth(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep_prob = 1 - self.p
        mask = torch.empty(x.size(0), 1, device=x.device).bernoulli_(keep_prob)
        return x / keep_prob * mask

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=384, num_classes=19, hidden_dim=128, dropout=0.3, stochastic_p=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.stochastic_depth = StochasticDepth(p=stochastic_p)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, label=None):
        x = self.proj(x)
        residual = self.residual(x)
        residual = self.stochastic_depth(residual)
        x = x + residual
        logits = self.classifier(x)
        return logits, x, None