import os
from pathlib import Path

class ModelConfig:
    GITHUB_REPO = "https://github.com/ywhhdj/emotion_classifier"
    GITHUB_TAG  = "v0.1.0"

    ONNX_FILES = ["emotion_classifier.onnx", "label_map.json"]
    PT_FILES   = ["emotion_classify.pt", "label_map.json"]

    CHECKSUMS = {
        "emotion_classify.pt": "sha256:d8e3c69a2d50f8c65100f4ca88aacbff4756410801f5da788850a7aa0d46b39d",
        "emotion_classifier.onnx": "sha256:85ff97e348eec0130b3d94113a4c69d28a2cb0ff9e91d3d888d477b85eeecaf9"
    }

    @classmethod
    def default_model_dir(cls) -> Path:
        env = os.environ.get("EMOTION_MODEL_DIR")
        if env:
            return Path(env)
        return Path.home() / ".emotion_classifier" / "models"

    @classmethod
    def download_url(cls, filename: str) -> str:
        return (
            f"{cls.GITHUB_REPO}/releases/download/"
            f"{cls.GITHUB_TAG}/{filename}"
        )

from torch.nn import Module
class EmotionClassifierNet(Module):
    """轻量多语言情感分类头。"""
    def __init__(self, input_dim=384, num_classes=19, hidden_dim=256, dropout=0.3):
        super().__init__()
        from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout
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