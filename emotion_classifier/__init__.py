"""emotion_classifier v0.2.0 —— 多语言情感分类（18 标签 / 中英日）。

推荐入口：

```python
from src import EmotionClassifier                 # 仓库内
from emotion_classifier import EmotionClassifier  # 安装后（别名包）
```
"""

from .predictor import EmotionClassifier
from .models.encoder import (
    BaseEncoder,
    ONNXEncoder,
    RustEngineEncoder,
    SentenceTransformerEncoder,
)
from .models.config import (
    MODEL_FILES,
    SUPPORTED_PRECISIONS,
    VERSION,
    ModelConfig,
    ModelRepository,
    model_filename,
    normalize_precision,
    precision_of,
)
from .models.loaded import LoadedModel
from .runtime import (
    attach_model,
    current_model,
    get_engine,
    release_model,
    select_backend,
)
from .languages import apply_lang_token, detect_lang, detect_langs, normalize_lang
from .threshold import Calibration, load_calib
from .exceptions import (
    EmotionClassifierError,
    ModelLoadError,
    InvalidInputError,
    BackendError,
    TokenizerError,
)

__version__ = VERSION

__all__ = [
    "EmotionClassifier",
    "LoadedModel",
    "attach_model",
    "release_model",
    "current_model",
    "get_engine",
    "select_backend",
    "BaseEncoder",
    "ONNXEncoder",
    "RustEngineEncoder",
    "SentenceTransformerEncoder",
    "ModelConfig",
    "ModelRepository",
    "MODEL_FILES",
    "SUPPORTED_PRECISIONS",
    "model_filename",
    "normalize_precision",
    "precision_of",
    "Calibration",
    "load_calib",
    "detect_lang",
    "detect_langs",
    "normalize_lang",
    "apply_lang_token",
    "EmotionClassifierError",
    "ModelLoadError",
    "InvalidInputError",
    "BackendError",
    "TokenizerError",
    "VERSION",
    "__version__",
]
