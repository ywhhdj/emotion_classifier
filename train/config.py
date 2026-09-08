import os
from pathlib import Path
from dotenv import load_dotenv

from .models import PoolingStrategy

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config:

    # ── 路径 ──
    @staticmethod
    def get_base_path(relative_path: str | Path) -> str:
        import sys
        base = getattr(sys, "_MEIPASS", None) or str(_PROJECT_ROOT)
        return os.path.normpath(os.path.join(base, str(relative_path)))

    @classmethod
    def onnx_encoder_path(cls) -> str:
        return os.environ.get("EMO_ONNX_ENCODER", cls.get_base_path("models/model_quint8_avx2.onnx"))

    @classmethod
    def onnx_encoder_tokenizer_dir(cls) -> str:
        return os.environ.get("EMO_ONNX_TOKENIZER", cls.get_base_path("models/tokenizer"))

    @classmethod
    def output_path(cls) -> str:
        return os.environ.get("EMO_OUTPUT", cls.get_base_path("output"))

    @classmethod
    def data_path(cls) -> str:
        return os.environ.get("EMO_DATA", cls.get_base_path("data/emotion.csv"))

    @classmethod
    def embed_path(cls) -> str:
        return os.environ.get("EMO_EMBED", cls.get_base_path("embeddings.npz"))

    @classmethod
    def model_path(cls) -> str:
        return os.environ.get("EMO_MODEL", cls.get_base_path("model/emotion_classify.pt"))

    @classmethod
    def onnx_path(cls) -> str:
        return os.environ.get("EMO_ONNX", cls.get_base_path("model/emotion_classifier.onnx"))

    @classmethod
    def onnx_quant_path(cls) -> str:
        return os.environ.get("EMO_ONNX_QUANT", cls.get_base_path("model/emotion_classifier_int8.onnx"))

    @classmethod
    def onnx_fp16_path(cls) -> str:
        return os.environ.get("EMO_ONNX_FP16", cls.get_base_path("model/emotion_classifier_fp16.onnx"))

    @classmethod
    def labelmap_path(cls) -> str:
        return os.environ.get("EMO_LABEL", cls.get_base_path("label_map.json"))

    @classmethod
    def model_cache_dir(cls) -> str:
        return os.environ.get("EMO_CACHE", cls.get_base_path("models"))

    # 运行期可被子命令覆盖（保持向后兼容的写法：直接赋值类属性即可）
    OUTPUT_PATH = output_path.__func__(object) if False else None  # 占位，实际见 __init__
    DATA_PATH = None
    EMBED_PATH = None
    MODEL_PATH = None
    ONNX_PATH = None
    ONNX_QUANT_PATH = None
    ONNX_FP16_PATH = None
    LABELMAP_PATH = None
    MODEL_CACHE_DIR = None

    MODEL_AUTO_DOWNLOAD = True

    # ── 模型 ──
    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    INPUT_DIM = 384
    HIDDEN_DIM = 128
    DROPOUT = 0.3
    NUM_LABELS = 18          # 仅作兜底，真实值由数据推断
    STOCHASTIC_P = 0.2

    # ── 训练参数 ──
    FOCAL_GAMMA = 2.0
    USE_FOCAL_LOSS = True
    LABEL_SMOOTHING = 0.05
    CLASS_WEIGHT = "effective"   # none | inverse | sqrt | effective
    EFFECTIVE_BETA = 0.999
    MIXUP_ALPHA = 0.2
    TEST_SIZE = 0.15
    VAL_SIZE = 0.10
    RANDOM_STATE = 42
    BATCH_SIZE = 64
    EPOCHS = 120
    LR = 5e-4
    WEIGHT_DECAY = 1e-2
    GRAD_CLIP = 1.0
    EARLY_STOP = 40
    METRIC = "macro_f1"      # accuracy | macro_f1（类别不平衡下 accuracy 会误导早停）

    # 余弦热重启调度
    SCHED_T0 = 10
    SCHED_T_MULT = 2
    SCHED_ETA_MIN = 1e-6

    # EMA
    EMA_DECAY = 0.999
    EMA_WARMUP_EPOCHS = 3

    # 推理
    MAX_SEQ_LEN = 128
    POOL_STRATEGY = PoolingStrategy.MEAN

    USE_MULTILINGUAL_AUG = True
    MINORITY_THRESHOLD = 0.8      # 少于「均值的该比例」视为少数类
    AUG_FACTOR = 2

    NUM_WORKERS = 0               # 数据已在内存，多进程反而更慢
    SEED = 42
    WRITE_FILE = True

    # ── 运行模式（惰性，避免 import 即加载 torch）──
    _DEVICE: str | None = None

    @classmethod
    def device(cls) -> str:
        if cls._DEVICE is None:
            try:
                import torch
                cls._DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                cls._DEVICE = "cpu"
        return cls._DEVICE

    @classmethod
    def refresh_paths(cls) -> None:
        """把 *_PATH 类属性同步为当前 env / 默认值（供 CLI 覆盖后调用）。"""
        cls.OUTPUT_PATH = cls.output_path()
        cls.DATA_PATH = cls.data_path()
        cls.EMBED_PATH = cls.embed_path()
        cls.MODEL_PATH = cls.model_path()
        cls.ONNX_PATH = cls.onnx_path()
        cls.ONNX_QUANT_PATH = cls.onnx_quant_path()
        cls.ONNX_FP16_PATH = cls.onnx_fp16_path()
        cls.LABELMAP_PATH = cls.labelmap_path()
        cls.MODEL_CACHE_DIR = cls.model_cache_dir()

    @classmethod
    def ensure_dirs(cls) -> None:
        os.makedirs(cls.OUTPUT_PATH, exist_ok=True) # type :ignore
        os.makedirs(cls.MODEL_CACHE_DIR, exist_ok=True) # type :ignore
        os.makedirs(os.path.dirname(cls.MODEL_PATH), exist_ok=True) # type :ignore
        os.makedirs(os.path.dirname(cls.ONNX_PATH), exist_ok=True) # type :ignore


Config.refresh_paths()