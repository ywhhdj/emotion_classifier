"""全局配置（v2.0：编码器由 MiniLM 换为 XLM-RoBERTa-base）。

与 v1 的 breaking 变化：
  * MODEL_NAME: paraphrase-multilingual-MiniLM-L12-v2 -> xlm-roberta-base
  * INPUT_DIM : 384 -> 768
  * MAX_SEQ_LEN: 128 -> 48（数据 P99=45 字符，极短文本场景）
  * 新增 LoRA / 回译 / 跨语言扩充 / 三语词表裁剪配置段

旧行为通过 TRAIN_MODE="cached" + LEGACY_INPUT_DIM 保留，见 train/train.py。
"""

class PoolingStrategy:
    CLS = "cls"
    MEAN = "mean"
    MAX = "max"
    WEIGHTED = "weighted"
    CLS_MEAN = "cls+mean"

import os
from pathlib import Path
from dotenv import load_dotenv

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
        return os.environ.get("EMO_ONNX_ENCODER", cls.get_base_path("models/xlmr_encoder.onnx"))

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

    @classmethod
    def base_model_dir(cls) -> str:
        """训练基座目录（HF 格式，三语裁剪产物）。"""
        return os.environ.get("EMO_BASE_MODEL", cls.get_base_path(cls.BASE_MODEL_DIR))

    @classmethod
    def deploy_dir(cls) -> str:
        """部署产物目录（自包含：ONNX + tokenizer + label_map + calib）。

        与训练基座目录是**两个不同的目录**：训练侧用 HF 权重目录，
        src/ 与 Rust 内核只认这个部署目录。
        """
        return os.environ.get("EMO_DEPLOY_DIR", cls.get_base_path("models/deploy"))

    @classmethod
    def train_output_dir(cls) -> str:
        """训练产物中转目录（.pt / 中转 ONNX / HF best）。"""
        return os.environ.get("EMO_TRAIN_OUT", cls.get_base_path("model"))

    @classmethod
    def onnx_encoder_cache(cls) -> str:
        """cached 模式预编码用的编码器 ONNX（仅加速用，不是部署产物）。"""
        return os.environ.get("EMO_ONNX_ENCODER_CACHE",
                              cls.get_base_path("models/xlmr_encoder_cache.onnx"))

    @classmethod
    def calib_path(cls) -> str:
        """温度 / per-class 阈值校准文件。"""
        return os.environ.get("EMO_CALIB", cls.get_base_path("model/calib.json"))

    @classmethod
    def bt_cache_path(cls) -> str:
        """回译结果缓存（只跑一次，CI 只读）。"""
        return os.environ.get("EMO_BT_CACHE", cls.get_base_path("data/bt_cache.jsonl"))

    # 运行期可被子命令覆盖
    OUTPUT_PATH = None
    DATA_PATH = None
    EMBED_PATH = None
    MODEL_PATH = None
    ONNX_PATH = None
    ONNX_QUANT_PATH = None
    ONNX_FP16_PATH = None
    LABELMAP_PATH = None
    MODEL_CACHE_DIR = None
    CALIB_PATH = None
    BT_CACHE_PATH = None

    MODEL_AUTO_DOWNLOAD = True
    BASE_MODEL_DIR_RES = None      # 解析后的实际基座目录（由 refresh_paths 填）
    DEPLOY_DIR_RES = None

    # ── 编码器基座（训练侧：HuggingFace 格式目录，**不是 ONNX**）──
    # 三语词表裁剪的产物目录，内含 config.json / model.safetensors /
    # sentencepiece.bpe.model。训练一律从这里 from_pretrained。
    BASE_MODEL_DIR = "models/xlmr-zh-en-ja"
    # 仅当本地目录不存在时，回退到 hub 下载原始 XLM-R（250k 词表）
    HUB_MODEL_NAME = "xlm-roberta-base"
    FALLBACK_TO_HUB = True
    MODEL_NAME = HUB_MODEL_NAME   # refresh_paths() 里解析为「本地目录或 hub 名」
    MODEL_TYPE = "xlm-roberta"
    INPUT_DIM = 768                        # XLM-R hidden_size
    LEGACY_INPUT_DIM = 384                 # 旧 MiniLM 分类头的维度（仅回退用）
    HIDDEN_DIM = 128
    DROPOUT = 0.1
    NUM_LABELS = 18                        # 仅作兜底，真实值由数据推断
    STOCHASTIC_P = 0.2

    # 序列长度：数据 max=45 字符，48 足够且把计算量降到约 1/3
    MAX_SEQ_LEN = 64
    POOL_STRATEGY = PoolingStrategy.CLS
    USE_LANG_TOKEN = False                  # 输入前缀 <zh>/<en>/<ja>
    LANG_TOKENS = {"zh": "<zh>", "en": "<en>", "ja": "<ja>"}
    SUPPORTED_LANGS = ("zh", "en", "ja")

    # ── 训练模式 ──
    # e2e   : 端到端（LoRA / 全量），文本级 DataLoader，不使用 embeddings.npz
    # cached: 冻结编码器预编码（含 prompt 完形路径），沿用 v1 两段式
    TRAIN_MODE = "e2e"

    # ── LoRA（路径 A）──
    USE_LORA = True
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ("query", "value")   # XLM-R 用 query/key/value，不是 q_proj/v_proj
    LORA_UNFREEZE_FROM = 10                     # 只对 layer[10:11] 挂 LoRA
    LORA_LR = 5e-5
    HEAD_LR = 5e-4
    LLRD = 0.9                                 # layer-wise lr decay
    USE_LLRD = False

    # ── prompt 完形（路径 B）──
    USE_PROMPT = False
    PROMPT_TEMPLATES = {
        "zh": "这句话的情感是[MASK]。{text}",
        "en": "The emotion of this sentence is [MASK]. {text}",
        "ja": "この文の感情は[MASK]です。{text}",
    }
    PROMPT_MASK_TOKEN = "[MASK]"

    # ── 正则 ──
    R_DROP_ALPHA = 0.5                     # 0 关闭
    FGM_EPS = 0.5                          # 0 关闭，仅 e2e 生效
    EMA_DECAY = 0.999
    EMA_WARMUP_EPOCHS = 3

    # ── 训练参数 ──
    FOCAL_GAMMA = 2.0
    USE_FOCAL_LOSS = True
    LABEL_SMOOTHING = 0.05
    CLASS_WEIGHT = "effective"             # none | inverse | sqrt | effective
    EFFECTIVE_BETA = 0.999
    MIXUP_ALPHA = 0.0                      # v2：端到端下默认关闭（序列级 MixUp 需重实现）
    TEST_SIZE = 0.15
    VAL_SIZE = 0.10
    RANDOM_STATE = 42
    BATCH_SIZE = 32                        # e2e 用 32；cached 模式 CLI 可覆盖为 64
    EPOCHS = 40                            # e2e 只需 20 轮；cached 模式沿用 120
    CACHED_EPOCHS = 120
    LR = 1e-4                              # e2e 默认 LoRA lr；cached 用 5e-4
    CACHED_LR = 5e-4
    WEIGHT_DECAY = 5e-2
    GRAD_CLIP = 1.0
    EARLY_STOP = 10                         # e2e；cached 沿用 40
    CACHED_EARLY_STOP = 40
    METRIC = "macro_f1"

    # 余弦热重启调度
    SCHED_T0 = 20
    SCHED_T_MULT = 2
    SCHED_ETA_MIN = 1e-6

    # ── 增强 ──
    USE_MULTILINGUAL_AUG = True
    MINORITY_THRESHOLD = 0.8
    AUG_FACTOR = 2
    EDA_SAFE_LANGS = ("en",)               # 只有英文保留字符级 EDA（中日文会被破坏）
    EDA_PUNCT_ENABLED = False              # 标点替换会改变情感标签，默认关闭

    # 回译（NLLB）
    USE_BACKTRANS = False
    BT_MODEL = "facebook/nllb-200-distilled-600M"
    BT_PIVOT = "eng_Latn"
    BT_LANG_CODE = {"zh": "zho_Hans", "en": "eng_Latn", "ja": "jpn_Japa"}
    BT_PIVOT_SIDE = {"zh": "deu_Latn", "en": "deu_Latn", "ja": "jpn_Japa"}  # en 走 de/fr，见代码
    BT_FACTOR = 1                          # 每条原样本最多产出几条回译样本
    BT_MIN_JACCARD = 0.45                  # 过近=无增益
    BT_MAX_JACCARD = 0.95                  # 过远=语义漂移
    BT_MIN_LEN_RATIO = 0.6
    BT_MAX_LEN_RATIO = 1.7

    # 跨语言扩充（高资源 -> 低资源）
    USE_CROSSLING = False
    CL_TRANSFER = (("zh", "ja"), ("zh", "en"), ("en", "ja"), ("ja", "en"))
    CL_TARGET_QUOTA = {"zh": 1.0, "en": 3.0, "ja": 3.0}   # 相对现有量的目标倍数
    CL_MAX_RATIO = 1.0                                     # MT 样本不超过真实样本的 1 倍
    CL_SAMPLE_WEIGHT = 0.7                                 # MT 样本的损失权重

    # ── 轻量化 ──
    TRIM_VOCAB = False
    TRIM_MODE = "intersect"                # textpruner | intersect | hf-trim
    TARGET_VOCAB_SIZE = 40000
    VOCAB_MIN_FREQ = 2
    VOCAB_CORPUS = ""                      # 三语语料路径（txt，一行一条）
    DAPT_EPOCHS = 1
    DAPT_LR = 1e-4

    PRUNE_ENCODER = False
    TARGET_FFN_SIZE = 1536
    TARGET_NUM_HEADS = 8
    PRUNE_N_ITERS = 12
    PRUNE_MULTIPLE_OF = 64

    DISTILL_AFTER_PRUNE = True
    DISTILL_T = 3.0
    DISTILL_ALPHA = 1.0                    # CE
    DISTILL_BETA = 6.0                     # logits MSE
    DISTILL_GAMMA = 0.5                    # 中间层 Cos

    CALIBRATE = True                       # temperature + per-class 阈值

    NUM_WORKERS = 0
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

    # ── 基座解析 ──
    @staticmethod
    def is_complete_hf_dir(d: str) -> bool:
        """判定一个目录是否为可直接 from_pretrained 的 HF 模型目录。

        三件套缺一不可：config.json + 权重 + 分词模型。
        缺任何一个都不算「已裁剪就绪」，避免半截产物被误用。
        """
        if not d or not os.path.isdir(d):
            return False
        has_cfg = os.path.exists(os.path.join(d, "config.json"))
        has_w = any(os.path.exists(os.path.join(d, f))
                    for f in ("model.safetensors", "pytorch_model.bin",
                              "model.safetensors.index.json"))
        has_tok = any(os.path.exists(os.path.join(d, f))
                      for f in ("sentencepiece.bpe.model", "tokenizer.json",
                                "spiece.model", "source.spm"))
        return has_cfg and has_w and has_tok

    @classmethod
    def encoder_dir(cls) -> str:
        """训练侧实际使用的编码器路径：本地裁剪产物优先，否则 hub 名。"""
        local = cls.base_model_dir()
        if cls.is_complete_hf_dir(local):
            return local
        if cls.FALLBACK_TO_HUB:
            return cls.HUB_MODEL_NAME
        raise FileNotFoundError(
            f"训练基座目录不完整: {local}（需要 config.json + 权重 + 分词模型）。\n"
            f"请先跑: python scripts/vocab/trim_vocab.py --mode intersect --out {local}\n"
            f"或设置 FALLBACK_TO_HUB=False 前先把基座准备好。"
        )

    @classmethod
    def tokenizer_dir_for_train(cls) -> str:
        """训练用 tokenizer 目录：优先基座目录自带的。"""
        d = cls.encoder_dir()
        if os.path.isdir(d) and cls.is_complete_hf_dir(d):
            return d
        return cls.MODEL_CACHE_DIR or cls.get_base_path("models")

    @classmethod
    def is_trimmed(cls) -> bool:
        """当前基座是否为三语裁剪产物（决定 vocab size 与是否需要 DAPT）。"""
        return cls.is_complete_hf_dir(cls.base_model_dir())

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
        cls.CALIB_PATH = cls.calib_path()
        cls.BT_CACHE_PATH = cls.bt_cache_path()
        cls.BASE_MODEL_DIR_RES = cls.base_model_dir()
        cls.DEPLOY_DIR_RES = cls.deploy_dir()
        # MODEL_NAME 解析为「本地裁剪目录」或「hub 名」，供各处 from_pretrained 直接用
        cls.MODEL_NAME = cls.encoder_dir()

    @classmethod
    def ensure_dirs(cls) -> None:
        os.makedirs(cls.OUTPUT_PATH, exist_ok=True)  # type: ignore
        os.makedirs(cls.MODEL_CACHE_DIR, exist_ok=True)  # type: ignore
        os.makedirs(os.path.dirname(cls.MODEL_PATH), exist_ok=True)  # type: ignore
        os.makedirs(os.path.dirname(cls.ONNX_PATH), exist_ok=True)  # type: ignore


Config.refresh_paths()
