import hashlib
import os
import glob
from pathlib import Path
from typing import Optional, List, Dict

VERSION = "0.2.0"

# v2：整图优先（encoder+head 一个 ONNX），其次 fp16，最后 int8
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

    # v2 产物哈希需在发布时填入；留 None 表示不校验
    CHECKSUMS = {
        "emotion_classifier.onnx": None,
        "emotion_classifier_fp16.onnx": None,
        "emotion_classifier_int8.onnx": None,
        "emotion_classify.pt": None,
        "label_map.json": None,
        "calib.json": None,
    }

    def __init__(self, model_dir: str | Path, repo_url: Optional[str] = None, create: bool = True):
        self.model_dir = Path(model_dir)
        if create:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        self.repo_url = repo_url or self.DEFAULT_REPO_URL
        self._verify_cache: Dict[str, bool] = {}

    # ── 文件发现 ──
    def find_classifier_onnx(self) -> Optional[Path]:
        if not self.model_dir.exists():
            return None
        matches = [Path(p) for p in
                   glob.glob(str(self.model_dir / "emotion_classifier*.onnx"))]
        if not matches:
            return None
        matches.sort(key=_priority_of)
        return matches[0]

    def find_encoder_onnx(self) -> Optional[Path]:
        """v2：整图模式下不需要单独编码器；仅兼容旧产物。"""
        if not self.model_dir.exists():
            return None
        for name in ("xlmr_encoder.onnx", "model_quint8_avx2.onnx"):
            p = self.model_dir / name
            if p.exists():
                return p
        matches = sorted(glob.glob(str(self.model_dir / "model_*.onnx")))
        return Path(matches[0]) if matches else None

    def find_pytorch_weight(self) -> Optional[Path]:
        pt = self.model_dir / "emotion_classify.pt"
        return pt if pt.exists() else None

    def find_calib(self) -> Optional[Path]:
        p = self.model_dir / "calib.json"
        return p if p.exists() else None

    def find_tokenizer(self) -> Optional[Path]:
        """XLM-R tokenizer 目录（含 sentencepiece.bpe.model）。"""
        p = self.model_dir / "tokenizer"
        if p.is_dir() and any(f.startswith("tokenizer") or f.startswith("sentencepiece")
                              for f in os.listdir(p)):
            return p
        return None

    def is_whole_graph(self) -> bool:
        """整图产物：输入含 input_ids（而不是 384/768 维向量）。"""
        cls = self.find_classifier_onnx()
        if cls is None:
            return False
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(cls), providers=["CPUExecutionProvider"])
            return any(i.name == "input_ids" for i in sess.get_inputs())
        except Exception:
            return False

    # ── 部署目录校验 ──
    # 部署目录 = scripts/export/package_deploy.py 的产出，自包含、可离线。
    # 与训练侧的 HF 目录（models/xlmr-zh-en-ja）是两回事。
    DEPLOY_REQUIRED = ["label_map.json"]
    DEPLOY_ONNX = ["emotion_classifier.onnx", "emotion_classifier_fp16.onnx",
                   "emotion_classifier_int8.onnx"]
    DEPLOY_TOKENIZER_DIRS = ["tokenizer", "."]
    DEPLOY_TOKENIZER_FILES = ["sentencepiece.bpe.model", "spiece.model", "source.spm"]

    def validate_deploy_dir(self) -> Dict[str, bool]:
        """逐项检查部署目录完整性（缺什么一目了然）。"""
        d = self.model_dir
        res = {"label_map": (d / "label_map.json").exists()}
        res["onnx"] = any((d / n).exists() for n in self.DEPLOY_ONNX)
        res["tokenizer"] = any(
            (d / sub / f).exists()
            for sub in self.DEPLOY_TOKENIZER_DIRS for f in self.DEPLOY_TOKENIZER_FILES
        ) or (d / "tokenizer" / "tokenizer.json").exists()
        res["weights_for_candle"] = any(
            (d / n).exists() for n in ("model.safetensors", "pytorch_model.bin"))
        res["calib"] = (d / "calib.json").exists()
        return res

    def is_valid_deploy_dir(self) -> bool:
        v = self.validate_deploy_dir()
        return v["label_map"] and v["onnx"] and v["tokenizer"]

    def required_files(self, backend: str) -> List[str]:
        """部署目录所需文件（自包含，不再依赖任何 hub 下载）。"""
        need = ["label_map.json"]
        cls = self.find_classifier_onnx()
        if cls is not None:
            need.append(cls.name)
        elif self.find_pytorch_weight() is not None:
            need.append("emotion_classify.pt")
        else:
            need.append(self.DEPLOY_ONNX[0])
        # candle 后端还要权重；ort 后端只要 onnx
        if backend == "rust" and self.find_weight() is not None:
            need.append(self.find_weight().name)  # type: ignore
        return need

    def find_weight(self) -> Optional[Path]:
        for n in ("model.safetensors", "pytorch_model.bin"):
            p = self.model_dir / n
            if p.exists():
                return p
        return None

    # ── 校验 ──
    def verify_file(self, filename: str, use_cache: bool = True) -> bool:
        checksum = self.CHECKSUMS.get(filename)
        if checksum is None:
            return True
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
        # v2：整图产物走 release；不再从 sentence-transformers 拉 MiniLM 编码器
        return f"{self.repo_url}/{filename}"


class ModelConfig:
    VERSION = VERSION
    ENCODER_TOKENIZER = "xlm-roberta-base"
    ENCODER_NAME = "xlm-roberta-base"
    EXPECTED_DIM = 768            # XLM-R hidden_size（v1 MiniLM 为 384）
    CLASSIFIER_ONNX_DEFAULT = "emotion_classifier.onnx"
    LABEL_MAP_NAME = "label_map.json"
    CALIB_NAME = "calib.json"
    SUPPORTED_LANGS = ("zh", "en", "ja")
    LANG_TOKENS = {"zh": "<zh>", "en": "<en>", "ja": "<ja>"}
    MAX_SEQ_LEN = 48

    # 后端优先级：rust（内部内核） > onnx（回退） > pytorch
    ENCODER_BACKEND = os.environ.get("EMOTION_BACKEND", "rust")

    # ── 标签映射 ──
    @classmethod
    def candidate_label_map_paths(cls, model_dir: Optional[str | Path] = None) -> List[Path]:
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
