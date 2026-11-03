from .predictor import EmotionClassifier
from .models.encoder import (
    BaseEncoder,
    ONNXEncoder,
    RustEngineEncoder,
    SentenceTransformerEncoder,
)
from .models.config import VERSION, ModelConfig, ModelRepository
from .languages import detect_lang, detect_langs, normalize_lang
from .threshold import Calibration, load_calib
from .exceptions import (
    EmotionClassifierError,
    ModelLoadError,
    InvalidInputError,
    BackendError,
)

__version__ = VERSION

__all__ = [
    "EmotionClassifier",
    "BaseEncoder",
    "ONNXEncoder",
    "RustEngineEncoder",
    "SentenceTransformerEncoder",
    "ModelConfig",
    "ModelRepository",
    "Calibration",
    "load_calib",
    "detect_lang",
    "detect_langs",
    "normalize_lang",
    "EmotionClassifierError",
    "ModelLoadError",
    "InvalidInputError",
    "BackendError",
    "VERSION",
]
