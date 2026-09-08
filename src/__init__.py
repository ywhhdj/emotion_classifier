from .predictor import EmotionClassifier
from .models.encoder import ONNXEncoder, SentenceTransformerEncoder, BaseEncoder
from .models.config import VERSION, ModelConfig, ModelRepository

__version__ = VERSION

__all__ = [
    "EmotionClassifier",
    "ONNXEncoder",
    "SentenceTransformerEncoder",
    "BaseEncoder",
    "ModelConfig",
    "ModelRepository",
]