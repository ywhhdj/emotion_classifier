import os
import glob
from pathlib import Path
from typing import Optional, List

class ONNXEncoder:
    """
    用 ONNX Runtime 加载量化后的 SentenceTransformer 编码器。
    输入: 文本列表
    输出: 384 维句向量 (numpy float32)
    """
    def __init__(
            self,
            onnx_path: str,
            tokenizer_dir: Optional[str] = None,
            tokenizer_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        # 获取模型所有输入的元信息
        self.inputs_info = {i.name: i for i in self.sess.get_inputs()}
        self.output_name = self.sess.get_outputs()[0].name

        # 加载 tokenizer
        if tokenizer_dir and os.path.isdir(tokenizer_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir, 
                local_files_only=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode(self, texts, batch_size=32, normalize=True):
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch,
                padding=True, truncation=True, max_length=128,
                return_tensors="np"
            )
            feeds = {}
            for input_name in self.inputs_info:
                if input_name == "input_ids":
                    feeds[input_name] = enc["input_ids"].astype(np.int64)
                elif input_name == "attention_mask":
                    feeds[input_name] = enc["attention_mask"].astype(np.int64)
                elif input_name == "token_type_ids":
                    tti = enc.get("token_type_ids")
                    if tti is None:
                        tti = np.zeros_like(enc["input_ids"])
                    feeds[input_name] = tti.astype(np.int64)
                else:
                    if input_name in enc:
                        feeds[input_name] = enc[input_name].astype(np.int64)
                    else:
                        raise ValueError(f"ONNX 模型要求未知输入 '{input_name}'，请检查模型导出配置。")

            output = self.sess.run([self.output_name], feeds)[0]

            # 如果输出是三维（batch, seq_len, dim），做 mean pooling
            if output.ndim == 3: # type: ignore
                mask = enc["attention_mask"][:, :, None].astype(np.float32)
                output = (output * mask).sum(axis=1) / mask.sum(axis=1)

            all_embeddings.append(output.astype(np.float32)) # type: ignore

        embeddings = np.concatenate(all_embeddings, axis=0)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-9, None)
        return embeddings

class ModelConfig:
    GITHUB_REPO = "https://github.com/ywhhdj/emotion_classifier"
    GITHUB_TAG  = "v0.1.1"

    # ── 编码器配置 ──
    ENCODER_ONNX     = "model_quint8_avx2.onnx"
    ENCODER_TOKENIZER = "paraphrase-multilingual-MiniLM-L12-v2"
    ENCODER_ONNX_URL = (
        f"https://huggingface.co/sentence-transformers/"
        f"{ENCODER_TOKENIZER}/resolve/main/onnx/{ENCODER_ONNX}?download=true"
    )

    CLASSIFIER_ONNX_DEFAULT = "emotion_classifier.onnx"
    ENCODER_TOKENIZER_DIR = "tokenizer"
    # 支持的通配符模式（按优先级）
    CLASSIFIER_PATTERNS = [
        "emotion_classifier_fp16.onnx",
        "emotion_classifier.onnx",
        "emotion_classifier*.onnx",
    ]

    ONNX_FILES = [CLASSIFIER_ONNX_DEFAULT, ENCODER_ONNX]
    PT_FILES   = ["emotion_classify.pt", ENCODER_ONNX]

    CHECKSUMS = {
        "emotion_classify.pt": "sha256:79c5ed6bb90145b48ae766d763bb86f29c2dec4e147dbdcaa04f65823205e9ac",
        "emotion_classifier.onnx": "sha256:c170c1771c6b3f428f77da792605a834729304217fec39801d1adab19ba283cc",
        "emotion_classifier_fp16.onnx": "sha256:3115eeb73db3f4c2cbda454bf921982d6e04655d5adf41f85cdc1327c2910d75",
    }

    @classmethod
    def load_label_map(cls) -> Path:
        pkg_dir = Path(__file__).resolve().parent
        return pkg_dir/ ".." / 'data' / 'label_map.json'

    @classmethod
    def get_resource_path(cls, relative_path) -> Path:
        try:
            base_path = sys._MEIPASS # type: ignore
        except Exception:
            base_path = os.path.abspath(".")
        return Path(os.path.normpath(os.path.join(base_path, relative_path)))

    # ── 类方法 ──
    @classmethod
    def default_model_dir(cls) -> Path:
        env = os.environ.get("EMOTION_MODEL_DIR")
        if env:
            return Path(env)
        return cls.get_resource_path("models")

    @classmethod
    def download_url(cls, filename: str) -> str:
        """根据文件名返回对应的下载 URL。"""
        if filename == cls.ENCODER_ONNX:
            return cls.ENCODER_ONNX_URL
        return f"{cls.GITHUB_REPO}/releases/download/{cls.GITHUB_TAG}/{filename}"

    @classmethod
    def find_classifier_onnx(cls, model_dir: str | Path) -> Optional[Path]:
        """
        在 model_dir 中按优先级查找分类头 ONNX 文件。
        匹配模式: emotion_classifier.onnx > fp16 > int8 > 其他*

        Returns
        -------
        Optional[Path]
            找到的文件路径；未找到返回 None
        """
        model_dir = Path(model_dir)
        if not model_dir.exists():
            return None

        # 定义优先级排序
        def _priority(filepath: str) -> int:
            name = os.path.basename(filepath).lower()
            if name == "emotion_classifier.onnx":
                return 0   # 原始 FP32 最优先
            elif "fp16" in name:
                return 1   # FP16 次之
            elif "int8" in name or "quant" in name:
                return 2   # INT8 量化再次
            else:
                return 3   # 其他变体兜底

        # 先尝试精确匹配
        for pat in cls.CLASSIFIER_PATTERNS:
            if "*" not in pat:
                fp = model_dir / pat
                if fp.exists():
                    return fp

        # 再尝试通配符
        all_matches = []
        for pat in cls.CLASSIFIER_PATTERNS:
            if "*" in pat:
                matches = glob.glob(str(model_dir / pat))
                all_matches.extend(matches)

        if not all_matches:
            return None

        all_matches.sort(key=_priority)
        return Path(all_matches[0])

    @classmethod
    def list_classifier_onnx(cls, model_dir: str | Path) -> List[Path]:
        """
        列出 model_dir 中所有分类头 ONNX 变体（按优先级排序）。
        """
        model_dir = Path(model_dir)
        if not model_dir.exists():
            return []

        def _priority(filepath: str) -> int:
            name = os.path.basename(filepath).lower()
            if name == "emotion_classifier.onnx":
                return 0
            elif "fp16" in name:
                return 1
            elif "int8" in name or "quant" in name:
                return 2
            else:
                return 3

        all_files = glob.glob(str(model_dir / "emotion_classifier*.onnx"))
        all_files.sort(key=_priority)
        return [Path(f) for f in all_files]
