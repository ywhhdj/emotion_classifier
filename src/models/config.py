import hashlib
import os
import glob
from pathlib import Path
from typing import Optional, List, Dict

VERSION = "0.1.1"

_CLASSIFIER_PRIORITY = ["emotion_classifier.onnx", "emotion_classifier_fp16.onnx"]


def _priority_of(path: Path) -> int:
    """分类头优先级：原始 fp32 > fp16 > int8/quant > 其他。"""
    name = path.name.lower()
    if name == "emotion_classifier.onnx":
        return 0
    if "fp16" in name:
        return 1
    if "int8" in name or "quant" in name:
        return 2
    return 3


class ModelRepository:
    DEFAULT_REPO_URL = f"https://github.com/ywhhdj/emotion_classifier/releases/download/v{VERSION}"

    CHECKSUMS = {
        "emotion_classifier.onnx": "sha256:c170c1771c6b3f428f77da792605a834729304217fec39801d1adab19ba283cc",
        "emotion_classifier_fp16.onnx": "sha256:3115eeb73db3f4c2cbda454bf921982d6e04655d5adf41f85cdc1327c2910d75",
        "emotion_classify.pt": "sha256:79c5ed6bb90145b48ae766d763bb86f29c2dec4e147dbdcaa04f65823205e9ac",
        "model_quint8_avx2.onnx": None,  # 编码器来自 HuggingFace，不校验
    }

    def __init__(self, model_dir: str | Path, repo_url: Optional[str] = None, create: bool = True):
        self.model_dir = Path(model_dir)
        if create:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        self.repo_url = repo_url or self.DEFAULT_REPO_URL
        self._verify_cache: Dict[str, bool] = {}

    # ── 文件发现 ──
    def find_classifier_onnx(self) -> Optional[Path]:
        """按优先级查找分类头 ONNX 文件：原始 > fp16 > int8 > 其他"""
        if not self.model_dir.exists():
            return None
        matches = [Path(p) for p in
                   glob.glob(str(self.model_dir / "emotion_classifier*.onnx"))]
        if not matches:
            return None
        matches.sort(key=_priority_of)
        return matches[0]

    def find_encoder_onnx(self) -> Optional[Path]:
        if not self.model_dir.exists():
            return None
        enc = self.model_dir / "model_quint8_avx2.onnx"
        if enc.exists():
            return enc
        matches = sorted(glob.glob(str(self.model_dir / "model_*.onnx")))
        return Path(matches[0]) if matches else None

    def find_pytorch_weight(self) -> Optional[Path]:
        pt = self.model_dir / "emotion_classify.pt"
        return pt if pt.exists() else None

    def required_files(self, backend: str) -> List[str]:
        """返回当前后端真正需要的文件名（不存在时回退到默认名）。"""
        enc = self.find_encoder_onnx()
        enc_name = enc.name if enc else "model_quint8_avx2.onnx"

        if backend == "pytorch":
            return ["emotion_classify.pt", enc_name]

        cls = self.find_classifier_onnx()
        cls_name = cls.name if cls else "emotion_classifier.onnx"
        if backend == "onnx":
            return [cls_name, enc_name]

        # auto：优先 onnx，其次 pytorch，都没有则按 onnx 下载
        if cls is not None:
            return [cls_name, enc_name]
        if self.find_pytorch_weight() is not None:
            return ["emotion_classify.pt", enc_name]
        return ["emotion_classifier.onnx", enc_name]

    # ── 校验 ──
    def verify_file(self, filename: str, use_cache: bool = True) -> bool:
        checksum = self.CHECKSUMS.get(filename)
        if checksum is None:
            return True  # 无校验码视为通过
        if use_cache and filename in self._verify_cache:
            return self._verify_cache[filename]
        fp = self.model_dir / filename
        ok = fp.exists() and self.sha256(fp) == checksum.split(":")[1]
        if use_cache:
            self._verify_cache[filename] = ok
        return ok

    @staticmethod
    def sha256(filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def download_url(self, filename: str) -> str:
        if filename.startswith("model_") and filename.endswith(".onnx"):
            return (
                "https://huggingface.co/sentence-transformers/"
                "paraphrase-multilingual-MiniLM-L12-v2/resolve/main/onnx/"
                f"{filename}?download=true"
            )
        return f"{self.repo_url}/{filename}"


class ModelConfig:
    ENCODER_TOKENIZER = "paraphrase-multilingual-MiniLM-L12-v2"
    CLASSIFIER_ONNX_DEFAULT = "emotion_classifier.onnx"
    LABEL_MAP_NAME = "label_map.json"

    # ── 标签映射 ──
    @classmethod
    def candidate_label_map_paths(cls, model_dir: Optional[str | Path] = None) -> List[Path]:
        """
        label_map.json 的候选位置（按优先级）。
        原实现只认 包目录/../data/label_map.json，而训练产出在项目根 label_map.json，
        导致 --labels / 推理必然 FileNotFoundError。
        """
        cands: List[Path] = []
        env = os.environ.get("EMOTION_LABEL_MAP")
        if env:
            cands.append(Path(env))
        if model_dir:
            cands.append(Path(model_dir) / cls.LABEL_MAP_NAME)
        pkg_dir = Path(__file__).resolve().parent.parent  # src/
        cands += [
            pkg_dir / "data" / cls.LABEL_MAP_NAME,
            pkg_dir.parent / "data" / cls.LABEL_MAP_NAME,
            pkg_dir.parent / cls.LABEL_MAP_NAME,
            cls.default_model_dir() / cls.LABEL_MAP_NAME,
            Path.home() / ".emotion_classifier" / cls.LABEL_MAP_NAME,
        ]
        seen, out = set(), []
        for c in cands:
            c = c.expanduser()
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    @classmethod
    def get_label_map_path(cls, model_dir: Optional[str | Path] = None) -> Path:
        """返回第一个存在的候选路径；都不存在时返回优先级最高的候选（便于报错提示）。"""
        cands = cls.candidate_label_map_paths(model_dir)
        for c in cands:
            if c.exists():
                return c
        return cands[0]

    @classmethod
    def load_label_map(cls, model_dir: Optional[str | Path] = None) -> dict:
        path = cls.get_label_map_path(model_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"标签映射文件不存在，已尝试: {[str(c) for c in cls.candidate_label_map_paths(model_dir)]}"
            )
        with open(path, "r", encoding="utf-8") as f:
            import json
            return json.load(f)

    # ── 资源路径 ──
    @classmethod
    def get_resource_path(cls, relative_path: str | Path) -> Path:
        """
        原实现用 os.path.abspath(".")，默认模型目录随当前工作目录漂移。
        改为：MEIPASS > 环境变量 > 用户目录，保证跨 cwd 稳定。
        """
        base = getattr(__import__("sys"), "_MEIPASS", None)
        if base:
            return Path(os.path.normpath(os.path.join(base, relative_path)))
        return Path.home() / ".emotion_classifier" / relative_path

    @classmethod
    def default_model_dir(cls) -> Path:
        env = os.environ.get("EMOTION_MODEL_DIR")
        if env:
            return Path(env).expanduser()
        return cls.get_resource_path("models")