import os
from models import PoolingStrategy
import sys
from dotenv import load_dotenv
load_dotenv()

class Config:
    @staticmethod
    def get_base_path(relative_path:str):
        try:
            base = sys._MEIPASS #type: ignore
        except AttributeError:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, relative_path)

    OUTPUT_PATH   = os.environ.get("EMO_OUTPUT", get_base_path("output"))
    DATA_PATH     = os.environ.get("EMO_DATA", get_base_path("data/emotion.csv"))
    EMBED_PATH    = os.environ.get("EMO_EMBED", get_base_path("embeddings.npz"))
    MODEL_PATH    = os.environ.get("EMO_MODEL", get_base_path("model/emotion_classify.pt"))
    ONNX_PATH     = os.environ.get("EMO_ONNX",  get_base_path("model/emotion_classifier.onnx"))
    LABELMAP_PATH = os.environ.get("EMO_LABEL", get_base_path("label_map.json"))

    MODEL_CACHE_DIR   = os.environ.get("EMO_CACHE", get_base_path("models"))
    MODEL_AUTO_DOWNLOAD = True

    # 模型 
    MODEL_NAME   = "paraphrase-multilingual-MiniLM-L12-v2"
    INPUT_DIM    = 384
    HIDDEN_DIM   = 256 #256
    DROPOUT      = 0.6 #0.5
    NUM_LABELS   = 19          # 仅作默认/校验，实际以数据为准

    # 训练参数 
    TEST_SIZE     = 0.2
    VAL_SIZE      = 0.10
    RANDOM_STATE  = 42
    BATCH_SIZE    = 128
    EPOCHS        = 60 
    LR            = 1e-3
    WEIGHT_DECAY  = 1e-3
    WARMUP_RATIO  = 0.05 # 学习率预热比例
    GRAD_CLIP     = 1.0
    EARLY_STOP    = 15   # 验证集连续 N 轮不提升则停止
    FOCAL_GAMMA   = 2.0
    USE_FOCAL_LOSS = True # 是否使用类别权重
    TEST_SIZE     = 0.15
    RANDOM_STATE  = 42
    BATCH_SIZE    = 128

    # 推理 
    MAX_SEQ_LEN   = 128        # MiniLM 最大长度
    POOL_STRATEGY = PoolingStrategy.MEAN      # mean / max / weighted


    USE_MULTILINGUAL_AUG = True   # 是否注入英/日增强样本

    # 运行模式 
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    NUM_WORKERS = 0               # DataLoader workers（CPU 上 0 更安全）
    SEED = 42
    WRITE_FILE = True   # 是否将推理结果写入文件

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    

