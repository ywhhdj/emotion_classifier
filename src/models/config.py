import hashlib
import os
import glob
from pathlib import Path
from typing import Optional, List
VERSION = "0.1.1"

class ModelRepository:
    DEFAULT_REPO_URL = f"https://github.com/ywhhdj/emotion_classifier/releases/download/v{VERSION}"

    CHECKSUMS = {
        "emotion_classifier.onnx": "sha256:c170c1771c6b3f428f77da792605a834729304217fec39801d1adab19ba283cc",
        "emotion_classifier_fp16.onnx": "sha256:3115eeb73db3f4c2cbda454bf921982d6e04655d5adf41f85cdc1327c2910d75",
        "emotion_classify.pt": "sha256:79c5ed6bb90145b48ae766d763bb86f29c2dec4e147dbdcaa04f65823205e9ac",
        "model_quint8_avx2.onnx": None,  # 编码器来自 HuggingFace，不校验
    }

    def __init__(self, model_dir: str|Path, repo_url: Optional[str] = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.repo_url = repo_url or self.DEFAULT_REPO_URL


    def find_classifier_onnx(self) -> Optional[Path]:
        """按优先级查找分类头 ONNX 文件：原始 > fp16 > int8 > 其他"""
        if not self.model_dir.exists():
            return None
        # 精确匹配
        for name in ["emotion_classifier.onnx", "emotion_classifier_fp16.onnx"]:
            fp = self.model_dir / name
            if fp.exists():
                return fp
        # 通配符
        matches = sorted(glob.glob(str(self.model_dir / "emotion_classifier*.onnx")))
        if not matches:
            return None
        # 优先级排序
        def priority(p):
            name = Path(p).name.lower()
            if name == "emotion_classifier.onnx": return 0
            if "fp16" in name: return 1
            if "int8" in name or "quant" in name: return 2
            return 3
        matches.sort(key=priority)
        return Path(matches[0])

    def find_encoder_onnx(self) -> Optional[Path]:
        if not self.model_dir.exists():
            return None
        enc = self.model_dir / "model_quint8_avx2.onnx"
        if enc.exists():
            return enc
        matches = glob.glob(str(self.model_dir / "model_*.onnx"))
        if matches:
            return Path(matches[0])
        return None

    def find_pytorch_weight(self) -> Optional[Path]:
        pt = self.model_dir / "emotion_classify.pt"
        return pt if pt.exists() else None

    def required_files(self, backend: str) -> List[str]:
        if backend == "onnx":
            return ["emotion_classifier.onnx", "model_quint8_avx2.onnx"]
        elif backend == "pytorch":
            return ["emotion_classify.pt", "model_quint8_avx2.onnx"]
        else:  # auto
            if self.find_classifier_onnx():
                return ["emotion_classifier.onnx", "model_quint8_avx2.onnx"]
            elif self.find_pytorch_weight():
                return ["emotion_classify.pt", "model_quint8_avx2.onnx"]
            else:
                return ["emotion_classifier.onnx", "model_quint8_avx2.onnx"]

    def verify_file(self, filename: str) -> bool:
        checksum = self.CHECKSUMS.get(filename)
        if checksum is None:
            return True  # 无校验码视为通过
        fp = self.model_dir / filename
        if not fp.exists():
            return False
        expected_hash = checksum.split(":")[1]
        actual_hash = self.__sha256(fp)
        return actual_hash == expected_hash

    @staticmethod
    def __sha256(filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def download_url(self, filename: str) -> str:
        if filename == "model_quint8_avx2.onnx":
            return (
                "https://huggingface.co/sentence-transformers/"
                "paraphrase-multilingual-MiniLM-L12-v2/resolve/main/onnx/"
                "model_quint8_avx2.onnx?download=true"
            )
        return f"{self.repo_url}/{filename}"

class ModelConfig:
    ENCODER_TOKENIZER = "paraphrase-multilingual-MiniLM-L12-v2"
    CLASSIFIER_ONNX_DEFAULT = "emotion_classifier.onnx"

    @classmethod
    def get_label_map_path(cls) -> Path:
        pkg_dir = Path(__file__).resolve().parent
        return pkg_dir/ ".." / 'data' / 'label_map.json'

    @classmethod
    def get_resource_path(cls, relative_path) -> Path:
        try:
            base_path = sys._MEIPASS # type: ignore
        except Exception:
            base_path = os.path.abspath(".")
        return Path(os.path.normpath(os.path.join(base_path, relative_path)))

    @classmethod
    def default_model_dir(cls) -> Path:
        env = os.environ.get("EMOTION_MODEL_DIR")
        if env:
            return Path(env)
        return cls.get_resource_path("models")
