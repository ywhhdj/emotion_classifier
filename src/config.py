import os
from models import PoolingStrategy
import sys
from dotenv import load_dotenv
load_dotenv()

class Config:
    @staticmethod
    def get_base_path(relative_path: str):
        try:
            base = sys._MEIPASS  # type: ignore
        except AttributeError:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.normpath(os.path.join(base, relative_path))

    @classmethod
    @property
    def ONNX_ENCODER_PATH(cls):
        return os.environ.get("EMO_ONNX_ENCODER", cls.get_base_path("models/model_quint8_avx2.onnx"))

    @classmethod
    @property
    def ONNX_ENCODER_TOKENIZER_DIR(cls):
        return os.environ.get("EMO_ONNX_TOKENIZER", cls.get_base_path("models/tokenizer"))

    OUTPUT_PATH   = os.environ.get("EMO_OUTPUT", get_base_path("output"))
    DATA_PATH     = os.environ.get("EMO_DATA", get_base_path("data/emotion.csv"))
    EMBED_PATH    = os.environ.get("EMO_EMBED", get_base_path("embeddings.npz"))
    MODEL_PATH    = os.environ.get("EMO_MODEL", get_base_path("model/emotion_classify.pt"))
    ONNX_PATH     = os.environ.get("EMO_ONNX",  get_base_path("model/emotion_classifier.onnx"))
    ONNX_QUANT_PATH = os.environ.get("EMO_ONNX_QUANT", get_base_path("model/emotion_classifier_int8.onnx"))
    LABELMAP_PATH = os.environ.get("EMO_LABEL", get_base_path("label_map.json"))

    MODEL_CACHE_DIR   = os.environ.get("EMO_CACHE", get_base_path("models"))
    MODEL_AUTO_DOWNLOAD = True

    # 模型
    MODEL_NAME   = "paraphrase-multilingual-MiniLM-L12-v2"
    INPUT_DIM    = 384
    HIDDEN_DIM   = 128   # 从 256 降至 128，参数量减少 75%
    DROPOUT      = 0.3
    NUM_LABELS   = 18

    # 训练参数（已清理重复定义）
    FOCAL_GAMMA   = 2.0
    USE_FOCAL_LOSS = True
    TEST_SIZE     = 0.15
    VAL_SIZE      = 0.10
    RANDOM_STATE  = 42
    BATCH_SIZE    = 64
    EPOCHS        = 120
    LR            = 5e-4
    WEIGHT_DECAY  = 1e-2
    GRAD_CLIP     = 1.0
    EARLY_STOP    = 40

    # 余弦热重启调度
    SCHED_T0     = 10
    SCHED_T_MULT = 2
    SCHED_ETA_MIN = 1e-6

    # EMA
    EMA_DECAY    = 0.999

    # 推理
    MAX_SEQ_LEN   = 128
    POOL_STRATEGY = PoolingStrategy.MEAN

    USE_MULTILINGUAL_AUG = True

    # 运行模式
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    SEED = 42
    WRITE_FILE = True

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)