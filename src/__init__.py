from .predictor import EmotionClassifier
from .models.encoder import ONNXEncoder, SentenceTransformerEncoder
from .models.config import VERSION
__version__ = VERSION

__all__ = [
    "EmotionClassifier",
    "ONNXEncoder",
    "SentenceTransformerEncoder"
]