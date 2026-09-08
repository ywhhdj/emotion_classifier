import json
import logging
import sys
from pathlib import Path
from typing import Literal, Optional, Dict, List, Tuple, Union

import numpy as np

from .models.config import ModelConfig, ModelRepository
from .downloader import AsyncModelDownloader

logging.basicConfig(
    level=logging.INFO,
    format=" [%(levelname)s] : %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class EmotionClassifier:
    """
    多语言情感分类器（19 种情感，支持中英日）。

    Examples
    ------
    ```python
        clf = EmotionClassifier(auto_download=True)
        result = clf("今天心情真好")  # [('高兴', 0.95), ...]

        with EmotionClassifier() as clf:
            results = clf.predict(["text1", "text2"], top_k=5)
    ```
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        backend: str = "auto",
        encoder_path: Optional[str] = None,
        auto_download: bool = False,
        device: Optional[Literal["cpu", "cuda"]] = None,
        tokenizer_dir: Optional[str] = None,
        repo_url: Optional[str] = None,
        encoder_cache_size: int = 0,
    ):
        self.model_dir = Path(model_dir) if model_dir else ModelConfig.default_model_dir()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.auto_download = auto_download
        self.encoder_path = encoder_path
        self.repo_url = repo_url
        # 修复：原实现仅在 tokenizer_dir is None 时赋值，传参时该属性根本不存在
        self.tokenizer_dir = str(tokenizer_dir) if tokenizer_dir else str(self.model_dir / "tokenizer")
        self._encoder_cache_size = int(encoder_cache_size)
        self.device = self._resolve_device(device)

        self._loaded = False
        self._label2id: Dict[str, int] = {}
        self._id2label: Dict[int, str] = {}
        self._num_labels: int = 0
        self._encoder = None
        self._session = None
        self._pt_model = None
        self._repo: Optional[ModelRepository] = None

    # ── 设备探测：不再为了判断 cuda 而 import torch（冷启动省 1~3s）──
    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device:
            return device
        env = (str(__import__("os").environ.get("EMOTION_DEVICE", "")) or "").lower()
        if env in ("cpu", "cuda"):
            return env
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    # ── 加载 ──
    def ensure_loaded(self):
        if self._loaded:
            return
        self.__initialize()
        self._loaded = True

    def __initialize(self):
        self._repo = ModelRepository(self.model_dir, repo_url=self.repo_url)
        needed = self._repo.required_files(self.backend)
        missing = [f for f in needed if not (self.model_dir / f).exists()]
        if missing:
            if self.auto_download:
                logger.info(f"[初始化] 检测到 {len(missing)} 个模型文件缺失，开始下载...")
                AsyncModelDownloader.download_sync(needed, self.model_dir, self._repo)
            else:
                raise FileNotFoundError(
                    f"模型文件缺失: {missing}\n"
                    f"请设置 auto_download=True 或手动放置到 {self.model_dir}"
                )

        self.__load_label_map()
        self.__init_encoder()
        self.__init_backend()

    def __init_encoder(self):
        if self.encoder_path:
            enc_path = Path(self.encoder_path)
            if not enc_path.exists():
                raise FileNotFoundError(f"自定义编码器不存在: {enc_path}")
            if enc_path.suffix == ".onnx":
                from .models.encoder import ONNXEncoder
                self._encoder = ONNXEncoder(
                    str(enc_path),
                    tokenizer_dir=self.tokenizer_dir,
                    cache_size=self._encoder_cache_size,
                )
            else:
                from .models.encoder import SentenceTransformerEncoder
                self._encoder = SentenceTransformerEncoder(str(enc_path), device=self.device)
            return

        enc_onnx = self._repo.find_encoder_onnx() #type: ignore
        if enc_onnx:
            from .models.encoder import ONNXEncoder
            self._encoder = ONNXEncoder(
                str(enc_onnx),
                tokenizer_dir=self.tokenizer_dir,
                cache_size=self._encoder_cache_size,
            )
            return

        logger.info("未找到 ONNX 编码器，回退到 SentenceTransformer...")
        from .models.encoder import SentenceTransformerEncoder
        self._encoder = SentenceTransformerEncoder(
            f"sentence-transformers/{ModelConfig.ENCODER_TOKENIZER}",
            device=self.device,
        )

    def __load_label_map(self):
        if self._label2id:
            return
        lm_path = ModelConfig.get_label_map_path(self.model_dir)
        if not lm_path.exists():
            raise FileNotFoundError(f"标签映射文件不存在: {lm_path}")
        with open(lm_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self._id2label = {int(k): v for k, v in mapping["id2label"].items()}
        self._label2id = mapping["label2id"]
        self._num_labels = int(mapping.get("num_labels", len(self._id2label)))

    def __init_backend(self):
        if self.backend == "auto":
            if self._repo.find_classifier_onnx() is not None:#type: ignore
                self.backend = "onnx"
            elif self._repo.find_pytorch_weight() is not None:#type: ignore
                self.backend = "pytorch"
            else:
                self.backend = "onnx"

        if self.backend == "onnx":
            self.__init_onnx()
        else:
            self.__init_pytorch()

    def __init_onnx(self):
        import onnxruntime as ort
        cls_path = self._repo.find_classifier_onnx()#type: ignore
        if cls_path is None:
            cls_path = self.model_dir / ModelConfig.CLASSIFIER_ONNX_DEFAULT
            if not cls_path.exists():
                # 有 pt 权重就优雅回退，而不是直接抛错
                if self._repo.find_pytorch_weight() is not None: #type: ignore
                    logger.warning("未找到分类头 ONNX，回退到 PyTorch 后端")
                    self.backend = "pytorch"
                    return self.__init_pytorch()
                raise FileNotFoundError(
                    f"未找到分类头 ONNX 文件，请确认 {self.model_dir} 中包含 emotion_classifier*.onnx"
                )
        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers.insert(0, "CUDAExecutionProvider")
        from .models.encoder import _default_session_options
        self._session = ort.InferenceSession(
            str(cls_path), sess_options=_default_session_options(), providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._input_dtype = self._session.get_inputs()[0].type
        logger.info(f"ONNX 后端就绪: {cls_path.name} (输入类型: {self._input_dtype})")

    def __init_pytorch(self):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch 后端需要安装 torch。pip install torch")
        from .models.pytorch import EmotionClassifier as PTModel

        pt_path = self._repo.find_pytorch_weight() #type: ignore
        if pt_path is None:
            raise FileNotFoundError(f"未找到 PyTorch 权重: {self.model_dir / 'emotion_classify.pt'}")

        self._pt_model = PTModel(
            input_dim=384,
            num_classes=self._num_labels or 19,
            hidden_dim=128,
            dropout=0.3,
        ).to(self.device)
        state = torch.load(pt_path, map_location=self.device)
        state = state.get("state_dict", state)
        self._pt_model.load_state_dict(state)
        self._pt_model.eval()
        logger.info(f"PyTorch 后端就绪: {pt_path.name}")

    # ── 推理 ──
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        self.ensure_loaded()
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return self._encoder.encode(texts, batch_size=batch_size) #type: ignore

    def predict(
        self,
        texts: Union[str, List[str]],
        top_k: int = 3,
        batch_size: int = 32,
    ) -> List[List[Tuple[str, float]]]:
        """情感分类预测，返回 [[(label, score), ...], ...]。"""
        return [self._to_pairs(p, top_k) for p in self.predict_proba(texts, batch_size)]

    def predict_proba(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """返回 (N, num_labels) 的概率矩阵，供评估/阈值调优复用。"""
        self.ensure_loaded()
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        texts = [t if isinstance(t, str) else "" for t in texts]
        if not texts:
            return np.zeros((0, self._num_labels), dtype=np.float32)

        embeddings = self.encode(texts, batch_size=batch_size)

        if self.backend == "onnx":
            x = embeddings.astype(np.float16) if self._input_dtype == "tensor(float16)" \
                else embeddings.astype(np.float32)
            logits = self._session.run(self._output_names, {self._input_name: x})[0] #type: ignore
        else:
            import torch
            with torch.no_grad():
                logits = self._pt_model(
                    torch.from_numpy(embeddings.astype(np.float32)).to(self.device)
                )[0].cpu().numpy() #type: ignore

        logits = np.asarray(logits, dtype=np.float32)
        logits -= logits.max(axis=1, keepdims=True)
        np.clip(logits, -60.0, None, out=logits)  # 防止 exp 下溢噪声
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum(axis=1, keepdims=True)

    def _to_pairs(self, prob_vec: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        k = max(1, min(int(top_k), len(prob_vec)))
        top_idx = np.argpartition(-prob_vec, k - 1)[:k]
        top_idx = top_idx[np.argsort(-prob_vec[top_idx])]
        return [(self._id2label.get(int(i), str(i)), float(round(float(prob_vec[i]), 4)))
                for i in top_idx]

    def predict_batch(self, texts: List[str], top_k: int = 3,
                      batch_size: int = 64) -> List[List[Tuple[str, float]]]:
        return self.predict(texts, top_k=top_k, batch_size=batch_size)

    def get_labels(self) -> List[str]:
        self.__load_label_map()
        return list(self._label2id.keys())

    def list_available_models(self) -> Dict[str, List[str]]:
        import glob
        result = {"classifier_onnx": [], "encoder_onnx": [], "pytorch": []}
        for f in glob.glob(str(self.model_dir / "emotion_classifier*.onnx")):
            result["classifier_onnx"].append(Path(f).name)
        for f in glob.glob(str(self.model_dir / "model_*.onnx")):
            result["encoder_onnx"].append(Path(f).name)
        pt = self.model_dir / "emotion_classify.pt"
        if pt.exists():
            result["pytorch"].append(pt.name)
        return result

    def update_models(self):
        """强制重新下载所有模型文件（修复：未加载时 self._repo 不存在）。"""
        self.close()
        repo = self._repo or ModelRepository(self.model_dir, repo_url=self.repo_url)
        self._repo = repo
        needed = repo.required_files(self.backend)
        for f in needed:
            fp = self.model_dir / f
            if fp.exists():
                fp.unlink()
        AsyncModelDownloader.download_sync(needed, self.model_dir, repo)
        self._loaded = False
        logger.info("模型更新完成")

    def model_info(self) -> Dict:
        # 修复：未 ensure_loaded 时访问 self._encoder 会 AttributeError
        if not self._loaded:
            self.__load_label_map()
        repo = self._repo or ModelRepository(self.model_dir, repo_url=self.repo_url)
        info = {
            "backend": self.backend,
            "model_dir": str(self.model_dir),
            "num_labels": self._num_labels,
            "labels": self.get_labels(),
            "encoder_type": type(self._encoder).__name__ if self._encoder else "(未加载)",
            "device": self.device,
        }
        info["files"] = {}
        for name in ["emotion_classifier.onnx", "emotion_classifier_fp16.onnx",
                     "emotion_classify.pt", "model_quint8_avx2.onnx", "label_map.json"]:
            fp = self.model_dir / name
            info["files"][name] = {
                "exists": fp.exists(),
                "size_mb": round(fp.stat().st_size / 1e6, 2) if fp.exists() else 0,
            }
        cls_path = repo.find_classifier_onnx()
        info["classifier"] = cls_path.name if cls_path else None
        return info

    def close(self):
        try:
            if getattr(self, "_encoder", None) is not None and hasattr(self._encoder, "close"):
                self._encoder.close() #type: ignore
        except Exception:
            pass
        self._encoder = None
        self._session = None
        self._pt_model = None
        self._loaded = False

    def __call__(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        return self.predict(text, top_k=top_k)[0]

    def __enter__(self):
        self.ensure_loaded()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass