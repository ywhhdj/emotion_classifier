import json
from pathlib import Path
import sys
from typing import Literal, Optional, Dict, List, Tuple, Union
import numpy as np
import logging

from .models.config import ModelConfig, ModelRepository
from .downloader import AsyncModelDownloader

logging.basicConfig(
    level=logging.INFO,
    format=" [%(levelname)s] : %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]  # 输出到 stdout，避免被重定向
)
logger = logging.getLogger(__name__)

class EmotionClassifier:
    """
    多语言情感分类器（19 种情感，支持中英日）。

    Examples
    ------
    ```python
        # 最简方式
        clf = EmotionClassifier(auto_download=True)
        result = clf("今天心情真好")  # [('高兴', 0.95), ...]

        # 自定义编码器
        clf = EmotionClassifier(encoder_path="/path/to/custom.onnx")

        # 上下文管理器
        with EmotionClassifier() as clf:
            results = clf.predict(["text1", "text2"], top_k=5)

        # 批量预测
        results = clf.predict_batch(texts, batch_size=64)
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
    ):
        self.model_dir = Path(model_dir) if model_dir else ModelConfig.default_model_dir()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.auto_download = auto_download
        self.encoder_path = encoder_path
        self.repo_url = repo_url
        if tokenizer_dir is None:
            self.tokenizer_dir = str(self.model_dir / "tokenizer")
        if device is None:
            try:
                from torch import cuda
                self.device = "cuda" if cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self._loaded = False
        self._label2id: Dict[str, int] = {}
        self._id2label: Dict[int, str] = {}
        self._num_labels: int = 0
    
    def ensure_loaded(self):
        if self._loaded:
            return
        self.__initialize()
        self._loaded = True
    
    def __initialize(self):
        self._repo = ModelRepository(self.model_dir, repo_url=self.repo_url)
        needed = self._repo.required_files(self.backend)
        missing = [f for f in needed if not (self.model_dir / f).exists()] # 检查缺失的模型文件
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
            # 用户指定了自定义编码器
            enc_path = Path(self.encoder_path)
            if not enc_path.exists():
                raise FileNotFoundError(f"自定义编码器不存在: {enc_path}")
            if enc_path.suffix == ".onnx":
                from .models.encoder import ONNXEncoder
                self._encoder = ONNXEncoder(str(enc_path), tokenizer_dir=self.tokenizer_dir)
            else:
                from .models.encoder import SentenceTransformerEncoder
                self._encoder = SentenceTransformerEncoder(str(enc_path), device=self.device)
            return

        # 自动查找 ONNX 编码器
        enc_onnx = self._repo.find_encoder_onnx()
        if enc_onnx:
            from .models.encoder import ONNXEncoder
            self._encoder = ONNXEncoder(str(enc_onnx), tokenizer_dir=self.tokenizer_dir)
            return

        # 回退到 SentenceTransformer
        logger.info("未找到 ONNX 编码器，回退到 SentenceTransformer...")
        from .models.encoder import SentenceTransformerEncoder
        self._encoder = SentenceTransformerEncoder(
            f"sentence-transformers/{ModelConfig.ENCODER_TOKENIZER}",
            device=self.device
        )

    def __load_label_map(self):
        if self._label2id:
            return
        lm_path = ModelConfig.get_label_map_path()
        if not lm_path.exists():
            raise FileNotFoundError(f"标签映射文件不存在: {lm_path}")
        with open(lm_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self._id2label = {int(k): v for k, v in mapping["id2label"].items()}
        self._label2id = mapping["label2id"]
        self._num_labels = mapping["num_labels"]

    def __init_backend(self):
        if self.backend == "auto":
            cls_onnx = self._repo.find_classifier_onnx()
            if cls_onnx:
                self.backend = "onnx"
            elif self._repo.find_pytorch_weight():
                self.backend = "pytorch"
            else:
                self.backend = "onnx"  # 默认

        if self.backend == "onnx":
            self.__init_onnx()
        else:
            self.__init_pytorch()
    
    def __init_onnx(self):
        import onnxruntime as ort
        cls_path = self._repo.find_classifier_onnx()
        if cls_path is None:
            # 尝试默认文件名
            cls_path = self.model_dir / ModelConfig.CLASSIFIER_ONNX_DEFAULT
            if not cls_path.exists():
                raise FileNotFoundError(
                    f"未找到分类头 ONNX 文件，请确认 {self.model_dir} 中包含 emotion_classifier*.onnx"
                )
        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            try:
                providers.insert(0, "CUDAExecutionProvider")
            except Exception:
                pass

        self._session = ort.InferenceSession(str(cls_path), providers=providers)
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

        pt_path = self._repo.find_pytorch_weight()
        if pt_path is None:
            raise FileNotFoundError(f"未找到 PyTorch 权重: {self.model_dir / 'emotion_classify.pt'}")

        self._pt_model = PTModel(
            input_dim=384,
            num_classes=self._num_labels,
            hidden_dim=128,
            dropout=0.3,
        ).to(self.device)
        state = torch.load(pt_path, map_location=self.device)
        self._pt_model.load_state_dict(state)
        self._pt_model.eval()
        logger.info(f"PyTorch 后端就绪: {pt_path.name}")

    def encode(self, texts: str|List[str], batch_size: int = 32) -> np.ndarray:
        self.ensure_loaded()
        if isinstance(texts, str):
            texts = [texts]
        return self._encoder.encode(texts, batch_size=batch_size)

    def predict(
        self,
        texts: Union[str, List[str]],
        top_k: int = 3,
        batch_size: int = 32,
    ) -> List[List[Tuple[str, float]]]:
        """
        情感分类预测。

        Parameters
        ----------
        texts : str or list of str
            待分类文本
        top_k : int
            返回前 k 个情感标签
        batch_size : int
            编码时的批次大小

        Returns
        -------
        results : list of dict
            每个元素为一个文本的预测结果，包含标签和概率
        """
        self.ensure_loaded()
        if isinstance(texts, str):
            texts = [texts]

        # 编码
        embeddings = self.encode(texts, batch_size=batch_size)

        # 推理
        if self.backend == "onnx":
            if self._input_dtype == "tensor(float16)":
                embeddings = embeddings.astype(np.float16)
            else:
                embeddings = embeddings.astype(np.float32)
            logits = self._session.run(self._output_names, {self._input_name: embeddings})[0] # type: ignore
        else:
            import torch
            with torch.no_grad():
                logits = self._pt_model(
                    torch.from_numpy(embeddings.astype(np.float32)).to(self.device)
                )[0].cpu().numpy()

        # 数值稳定 softmax
        logits = logits.astype(np.float32) # type: ignore
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        # 组装结果
        results = []
        for prob_vec in probs:
            top_idx = np.argsort(prob_vec)[::-1][:top_k]
            item = [(self._id2label[int(i)], float(round(prob_vec[i], 4))) for i in top_idx]
            results.append(item)
        return results
    
    def predict_batch(
        self,
        texts: List[str],
        top_k: int = 3,
        batch_size: int = 64,
    ) -> List[List[Tuple[str, float]]]:
        return self.predict(texts, top_k=top_k, batch_size=batch_size)

    def get_labels(self) -> List[str]:
        self.__load_label_map()
        return list(self._label2id.keys())

    def list_available_models(self) -> Dict[str, List[str]]:
        import glob
        result = {"classifier_onnx": [], "encoder_onnx": [], "pytorch": []}

        # 分类头
        for f in glob.glob(str(self.model_dir / "emotion_classifier*.onnx")):
            result["classifier_onnx"].append(Path(f).name)

        # 编码器
        for f in glob.glob(str(self.model_dir / "model_*.onnx")):
            result["encoder_onnx"].append(Path(f).name)

        # PyTorch
        pt = self.model_dir / "emotion_classify.pt"
        if pt.exists():
            result["pytorch"].append(pt.name)

        return result

    def update_models(self):
        """强制重新下载所有模型文件"""
        self.close()
        needed = self._repo.required_files(self.backend)
        for f in needed:
            fp = self.model_dir / f
            if fp.exists():
                fp.unlink()
        AsyncModelDownloader.download_sync(needed, self.model_dir, self._repo)
        self._loaded = False
        logger.info("模型更新完成")

    def model_info(self) -> Dict:
        info = {
            "backend": self.backend,
            "model_dir": str(self.model_dir),
            "num_labels": self._num_labels,
            "labels": self.get_labels(),
            "encoder_type": type(self._encoder).__name__,
        }
        # 文件状态
        info["files"] = {}
        for name in ["emotion_classifier.onnx", "emotion_classifier_fp16.onnx",
                      "emotion_classify.pt", "model_quint8_avx2.onnx"]:
            fp = self.model_dir / name
            info["files"][name] = {
                "exists": fp.exists(),
                "size_mb": round(fp.stat().st_size / 1e6, 2) if fp.exists() else 0,
            }
        return info

    
    def close(self):
        if hasattr(self, '_session') and self._session is not None:
            self._session = None
        if hasattr(self, '_pt_model'):
            del self._pt_model
        if hasattr(self, '_encoder') and hasattr(self._encoder, 'close'):
            self._encoder.close()
        self._loaded = False
    
    def __call__(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        return self.predict(text, top_k=top_k)[0]

    def __enter__(self):
        self.ensure_loaded()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()