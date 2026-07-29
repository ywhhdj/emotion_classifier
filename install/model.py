from torch.nn import Module, Sequential, Linear, BatchNorm1d, ReLU, Dropout
class EmotionClassifierNet(Module):
    """轻量多语言情感分类头。"""
    def __init__(self, input_dim=384, num_classes=19, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = Sequential(
            Linear(input_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, max(hidden_dim // 2, num_classes)),
            BatchNorm1d(max(hidden_dim // 2, num_classes)),
            ReLU(),
            Dropout(dropout),
            Linear(max(hidden_dim // 2, num_classes), num_classes),
        )

    def forward(self, x):
        return self.net(x)