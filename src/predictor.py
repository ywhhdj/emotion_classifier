"""对外预测器（v2.0）。

对外 API 与 v1 完全一致（predict / predict_batch / predict_proba / predict_label /
get_labels / model_info），内部改为：
  * 整图推理（encoder+head 一个模型，输入 input_ids/attention_mask）
  * Rust 内核优先（无 GIL、批调度在内核），ONNX / PyTorch 作为回退
  * 支持 calib.json 的温度与 per-class 阈值
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal, Optional, Dict, List, Tuple, Union

import numpy as np

from .models.config import ModelConfig, ModelRepository
from .downloader import AsyncModelDownloader
from .languages import detect_langs, normalize_langs
from .threshold import load_calib

logging.basicConfig(
    level=logging.INFO,
    format=" [%(levelname)s] : %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class EmotionClassifier:
    """多语言情感分类器（18 种情感，中文 / 英文 / 日文）。

    Examples
    ------
    ```python
        clf = EmotionClassifier(auto_download=True)
        clf.predict("今天心情真好")            # [[('高兴', 0.95), ...]]
        clf.predict_proba(["a", "b"])          # (N, 18)
        with EmotionClassifier() as clf:
            clf.predict(["text1", "text2"], top_k=5)
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
        num_threads: int = 4,
        max_batch: int = 32,
        use_lang_token: bool = True,
    ):
        self.model_dir = Path(model_dir) if model_dir else ModelConfig.default_model_dir()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.auto_download = auto_download
        self.encoder_path = encoder_path
        self.repo_url = repo_url
        self.tokenizer_dir = str(tokenizer_dir) if tokenizer_dir else str(self.model_dir / "tokenizer")
        self._encoder_cache_size = int(encoder_cache_size)
        self._num_threads = int(num_threads)
        self._max_batch = int(max_batch)
        self.use_lang_token = bool(use_lang_token)
        self.device = self._resolve_device(device)

        self._loaded = False
        self._label2id: Dict[str, int] = {}
        self._id2label: Dict[int, str] = {}
        self._num_labels: int = 0
        self._encoder = None
        self._session = None
        self._pt_model = None
        self._repo: Optional[ModelRepository] = None
        self._whole_graph = False
        self._tokenizer = None
        self._calib = None
        self._rust_engine = None

    # ── 设备探测 ──
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
        self._calib = load_calib(self.model_dir,
                                 __import__("os").environ.get("EMOTION_CALIB"))
        self._whole_graph = self._repo.is_whole_graph()
        self.__init_backend()

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
        # 1) Rust 内核（整图，一步到位）
        if self.backend in ("auto", "rust"):
            try:
                from .runtime import get_engine
                self._rust_engine = get_engine(
                    self.model_dir, num_threads=self._num_threads,
                    max_batch=self._max_batch, backend="candle", precision="fp16")
                self.backend = "rust"
                logger.info("Rust 内核就绪（candle / fp16）")
                return
            except Exception as e:
                if self.backend == "rust":
                    raise
                logger.info(f"Rust 内核不可用（{e}），回退到 ONNX Runtime")

        if self.backend == "auto":
            self.backend = "onnx" if self._repo.find_classifier_onnx() is not None else "pytorch"

        if self.backend == "onnx":
            self.__init_onnx()
        else:
            self.__init_pytorch()

    def __init_onnx(self):
        import onnxruntime as ort
        cls_path = self._repo.find_classifier_onnx()
        if cls_path is None:
            cls_path = self.model_dir / ModelConfig.CLASSIFIER_ONNX_DEFAULT
            if not cls_path.exists():
                if self._repo.find_pytorch_weight() is not None:
                    logger.warning("未找到分类头 ONNX，回退到 PyTorch 后端")
                    self.backend = "pytorch"
                    return self.__init_pytorch()
                raise FileNotFoundError(
                    f"未找到模型 ONNX 文件，请确认 {self.model_dir} 中包含 emotion_classifier*.onnx"
                )
        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers.insert(0, "CUDAExecutionProvider")
        from .models.encoder import _default_session_options
        self._session = ort.InferenceSession(
            str(cls_path), sess_options=_default_session_options(), providers=providers
        )
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._whole_graph = "input_ids" in self._input_names
        if self._whole_graph:
            self.__init_tokenizer()
        else:
            self.__init_encoder()
        logger.info(f"ONNX 后端就绪: {cls_path.name}（{'整图' if self._whole_graph else '编码器+头'}）")

    def __init_tokenizer(self):
        """部署目录自带 tokenizer，绝不在推理时联网拉 hub。"""
        from transformers import AutoTokenizer
        local = self._repo.find_tokenizer()
        env_dir = os.environ.get("EMOTION_TOKENIZER_DIR")
        for cand in (local, Path(env_dir) if env_dir else None):
            if cand is not None and Path(cand).is_dir():
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(str(cand),
                                                                   local_files_only=True)
                    return
                except Exception:
                    continue
        raise FileNotFoundError(
            f"部署目录缺少 tokenizer: {self.model_dir}（需要 tokenizer/ 或 "
            f"sentencepiece.bpe.model）\n"
            f"请先跑: python scripts/export/package_deploy.py "
            f"--model-dir model/pruned --out {self.model_dir}"
        )
        if self.use_lang_token:
            missing = [t for t in ModelConfig.LANG_TOKENS.values()
                       if t not in self._tokenizer.get_vocab()]
            if missing:
                logger.warning(f"词表缺少语言标记 {missing}，将不加 lang token")
                self.use_lang_token = False

    def __init_encoder(self):
        """非整图产物：单独的编码器 ONNX + 分类头。"""
        if self.encoder_path:
            from .models.encoder import ONNXEncoder
            self._encoder = ONNXEncoder(str(self.encoder_path),
                                        tokenizer_dir=self.tokenizer_dir,
                                        cache_size=self._encoder_cache_size,
                                        strict_dim=False)
            return
        enc_onnx = self._repo.find_encoder_onnx()
        if enc_onnx:
            from .models.encoder import ONNXEncoder
            self._encoder = ONNXEncoder(str(enc_onnx), tokenizer_dir=self.tokenizer_dir,
                                        cache_size=self._encoder_cache_size, strict_dim=False)
            return
        logger.info("未找到 ONNX 编码器，回退到 SentenceTransformer（v1 兼容）...")
        from .models.encoder import SentenceTransformerEncoder
        self._encoder = SentenceTransformerEncoder(device=self.device)

    def __init_pytorch(self):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch 后端需要安装 torch。pip install torch")
        from .models.pytorch import EmotionClassifier as PTModel

        pt_path = self._repo.find_pytorch_weight()
        if pt_path is None:
            raise FileNotFoundError(f"未找到 PyTorch 权重: {self.model_dir / 'emotion_classify.pt'}")

        ckpt = torch.load(pt_path, map_location=self.device)
        input_dim = int(ckpt.get("input_dim", ModelConfig.EXPECTED_DIM))
        hidden_dim = int(ckpt.get("hidden_dim", 128))
        self._pt_model = PTModel(input_dim=input_dim,
                                 num_classes=self._num_labels or 18,
                                 hidden_dim=hidden_dim, dropout=0.0).to(self.device)
        self._pt_model.load_state_dict(ckpt.get("state_dict", ckpt))
        self._pt_model.eval()
        logger.info(f"PyTorch 后端就绪: {pt_path.name}")

    # ── 推理 ──
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """返回 logits（整图）或句向量（旧两段式产物）。"""
        self.ensure_loaded()
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self._num_labels), dtype=np.float32)
        if self._rust_engine is not None:
            return np.asarray(self._rust_engine.forward(list(texts), None), dtype=np.float32)
        if self._whole_graph:
            return self.__onnx_logits(texts, batch_size)
        return self._encoder.encode(texts, batch_size=batch_size)  # type: ignore

    def __onnx_logits(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        max_len = ModelConfig.MAX_SEQ_LEN
        outs = []
        for s in range(0, len(texts), batch_size):
            chunk = list(texts[s:s + batch_size])
            langs = detect_langs(chunk)
            if self.use_lang_token:
                chunk = [f"{ModelConfig.LANG_TOKENS[l]}{t}" for l, t in zip(langs, chunk)]
            enc = self._tokenizer(chunk, padding=True, truncation=True,
                                  max_length=max_len, return_tensors="np")
            feeds = {}
            for name in self._input_names:
                if name in enc:
                    feeds[name] = enc[name].astype(np.int64)
                else:
                    feeds[name] = np.zeros_like(enc["input_ids"]).astype(np.int64)
            outs.append(self._session.run(self._output_names, feeds)[0])  # type: ignore
        return np.concatenate(outs, axis=0)

    def predict_proba(self, texts: Union[str, List[str]], batch_size: int = 32,
                      lang: Optional[Union[str, List[str]]] = None) -> np.ndarray:
        """返回 (N, num_labels) 的概率矩阵。"""
        self.ensure_loaded()
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        texts = [t if isinstance(t, str) else "" for t in texts]
        if not texts:
            return np.zeros((0, self._num_labels), dtype=np.float32)

        langs = normalize_langs(lang if isinstance(lang, list) else
                                ([lang] if lang else None), len(texts))
        if langs is None and self.use_lang_token:
            langs = detect_langs(texts)

        logits = np.asarray(self.encode(texts, batch_size), dtype=np.float32)

        # 温度校准
        if self._calib is not None:
            if langs:
                for i, lg in enumerate(langs):
                    logits[i] = self._calib.apply(logits[i], lg)
            else:
                logits = self._calib.apply(logits)

        logits -= logits.max(axis=1, keepdims=True)
        np.clip(logits, -60.0, None, out=logits)
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum(axis=1, keepdims=True)

    def predict(
        self,
        texts: Union[str, List[str]],
        top_k: int = 3,
        batch_size: int = 32,
        lang: Optional[Union[str, List[str]]] = None,
    ) -> List[List[Tuple[str, float]]]:
        """情感分类预测，返回 [[(label, score), ...], ...]。"""
        return [self._to_pairs(p, top_k)
                for p in self.predict_proba(texts, batch_size=batch_size, lang=lang)]

    def predict_label(self, texts: Union[str, List[str]],
                      lang: Optional[Union[str, List[str]]] = None
                      ) -> List[List[Tuple[str, float]]]:
        """只返回最可能的 1 个标签（README 中承诺的接口）。"""
        return self.predict(texts, top_k=1, lang=lang)

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
        result = {"classifier_onnx": [], "encoder_onnx": [], "pytorch": [], "calib": []}
        for f in glob.glob(str(self.model_dir / "emotion_classifier*.onnx")):
            result["classifier_onnx"].append(Path(f).name)
        for f in glob.glob(str(self.model_dir / "model_*.onnx")) + \
                glob.glob(str(self.model_dir / "xlmr_*.onnx")):
            result["encoder_onnx"].append(Path(f).name)
        if (self.model_dir / "emotion_classify.pt").exists():
            result["pytorch"].append("emotion_classify.pt")
        if (self.model_dir / "calib.json").exists():
            result["calib"].append("calib.json")
        return result

    def update_models(self):
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
        if not self._loaded:
            self.__load_label_map()
        repo = self._repo or ModelRepository(self.model_dir, repo_url=self.repo_url)
        info = {
            "version": ModelConfig.VERSION if hasattr(ModelConfig, "VERSION") else "0.2.0",
            "backend": self.backend,
            "encoder": ModelConfig.ENCODER_NAME,
            "whole_graph": self._whole_graph,
            "model_dir": str(self.model_dir),
            "num_labels": self._num_labels,
            "labels": self.get_labels(),
            "device": self.device,
            "calibrated": self._calib is not None and abs(self._calib.temperature - 1.0) > 1e-6,
        }
        info["files"] = {}
        for name in ["emotion_classifier.onnx", "emotion_classifier_fp16.onnx",
                     "emotion_classifier_int8.onnx", "emotion_classify.pt",
                     "label_map.json", "calib.json"]:
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
                self._encoder.close()  # type: ignore
        except Exception:
            pass
        # Rust 引擎由 runtime 单例持有，进程内复用，不在此释放
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
