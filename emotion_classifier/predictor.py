"""对外预测器

* 整图 ONNX 推理（encoder+head 一个模型，输入 input_ids/attention_mask）；
* 三档模型 `emotion_classifier[.|_fp16|_int8].onnx` 从 GitHub Release 下载；
* Rust 内核（`_core.EmotionEngine` + ort）优先，纯 Python ORT / PyTorch 回退；
* **支持传入已加载模型**（session / Rust engine / ONNX 字节 / 文件路径），
  模型生命周期可完全交给调用方；
* **进程内单一实例**：多个 EmotionClassifier 只初始化一次模型。
"""

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from .downloader import AsyncModelDownloader
from .exceptions import BackendError, InvalidInputError, ModelLoadError
from .languages import apply_lang_token, detect_lang, detect_langs, normalize_langs
from .models.config import (
    CALIB_NAME, MANIFEST_NAME, MODEL_FILES, ModelConfig, ModelRepository,
    normalize_precision, precision_of,
)
from .models.loaded import LoadedModel
from .runtime import (
    assert_compatible, attach_model, current_model, get_engine, release_model,
)
from .threshold import Calibration, load_calib

logging.basicConfig(
    level=logging.INFO,
    format=" [%(levelname)s] : %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class EmotionClassifier:
    """多语言情感分类器（18 种情感，中文 / 英文 / 日文）。

    Examples
    --------
    ```python
    from em import EmotionClassifier

    clf = EmotionClassifier(auto_download=True)     # 从 Release 下载三档模型
    clf.predict("今天心情真好", top_k=3)

    # 交给程序管理：注入自己已加载的模型
    import onnxruntime as ort
    sess = ort.InferenceSession("emotion_classifier_int8.onnx")
    clf2 = EmotionClassifier(model=sess, precision="int8")
    ```
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        backend: Literal["auto", "rust", "onnx", "pytorch"]= "auto",
        encoder_path: Optional[str] = None,
        auto_download: bool = False,
        device: Optional[Literal["cpu", "cuda"]] = None,
        tokenizer_dir: Optional[str] = None,
        repo_url: Optional[str] = None,
        encoder_cache_size: int = 0,
        num_threads: int = 4,
        max_batch: int = 32,
        use_lang_token: Optional[bool] = None,
        *,
        model: Any = None,
        model_path: Optional[str] = None,
        precision: Literal["auto", "fp32", "fp16", "int8"] = "auto",
        owns_model: bool = False,
        tokenizer: Any = None,
        label_map: Any = None,
        calib: Any = None,
        max_seq_len: Optional[int] = None,
        replace: bool = False,
    ):
        self.model_dir = Path(model_dir) if model_dir else ModelConfig.default_model_dir()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.auto_download = bool(auto_download)
        self.encoder_path = encoder_path
        self.repo_url = repo_url
        self.tokenizer_dir = str(tokenizer_dir) if tokenizer_dir else str(self.model_dir / "tokenizer")
        self._encoder_cache_size = int(encoder_cache_size)
        self._num_threads = int(num_threads)
        self._max_batch = int(max_batch)
        self.device = self._resolve_device(device)

        # 训练期参数：显式入参 > deploy_manifest.json > train/config.py 默认值
        self._explicit_use_lang_token = use_lang_token
        self.use_lang_token = bool(use_lang_token) if use_lang_token is not None \
            else ModelConfig.USE_LANG_TOKEN
        self._explicit_max_seq_len = max_seq_len
        self.max_seq_len = int(max_seq_len) if max_seq_len else ModelConfig.MAX_SEQ_LEN

        self.precision = normalize_precision(precision)
        self._precision = "fp32" if self.precision == "auto" else self.precision
        self._replace = bool(replace)
        self._owns_model = bool(owns_model)

        # 外部注入（程序管理模型）
        self._injected = model if model is not None else model_path
        self._injected_tokenizer = tokenizer
        self._injected_label_map = label_map
        self._injected_calib = calib

        self._load_lock = threading.RLock()
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
        self._calib: Optional[Calibration] = None
        self._rust_engine = None
        self._model: Optional[LoadedModel] = None
        self._model_source = "release"

    # ── 设备探测 ──
    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device:
            return device
        env = os.environ.get("EMOTION_DEVICE", "").lower()
        if env in ("cpu", "cuda"):
            return env
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    # ── 加载（线程安全、幂等） ──
    def ensure_loaded(self):
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            self.__initialize()
            self._loaded = True

    def __initialize(self):
        self._repo = ModelRepository(self.model_dir, repo_url=self.repo_url,tokenizer_dir=self.tokenizer_dir)
        meta = self._repo.get_deploy_meta()
        if self._explicit_use_lang_token is None and "use_lang_token" in meta:
            self.use_lang_token = bool(meta["use_lang_token"])
        if self._explicit_max_seq_len is None and meta.get("max_seq_len"):
            self.max_seq_len = int(meta["max_seq_len"])

        if self._injected is not None:
            self.__adopt_external()
            self.__finalize_metadata()
            return

        self.__ensure_files()
        cls = self._repo.find_classifier_onnx(self.precision)
        if cls is not None:
            self._precision = precision_of(cls)
        self.__init_backend()
        self.__finalize_metadata()

    def __finalize_metadata(self):
        """标签与校准：注入项优先，其次部署目录。"""
        injected = self._model.label_map if self._model is not None else None
        self.__load_label_map(injected or self._injected_label_map)
        calib = (self._model.calib if self._model is not None else None) \
            or self._injected_calib
        if calib is None:
            self._calib = load_calib(self.model_dir,
                                     os.environ.get("EMOTION_CALIB"))
        else:
            self._calib = self.__coerce_calib(calib)

    def __ensure_files(self):
        assert self._repo is not None
        needed = self._repo.required_files(self.precision)
        if self._injected is not None:
            needed = [f for f in needed if not f.endswith(".onnx")]
        missing = [f for f in needed if not (self.model_dir / f).exists()]
        # 已存在但校验和不匹配的资产也要重下
        for f in needed:
            if f not in missing and not self._repo.verify_file(f):
                missing.append(f)
        if missing:
            if self.auto_download:
                logger.info("[初始化] 检测到 %d 个文件缺失，开始从 Release 下载...", len(missing))
                AsyncModelDownloader.download_sync(
                    missing, self.model_dir, self._repo,
                    optional=self._repo.optional_files(),
                )
            else:
                raise FileNotFoundError(
                    f"模型文件缺失: {missing}\n"
                    f"请设置 auto_download=True 或手动放置到 {self.model_dir}\n"
                    f"Release: {self._repo.repo_url}"
                )

    def __adopt_external(self):
        """把调用方已加载的模型登记为进程内唯一实例。"""
        assert self._repo is not None
        # 注入模型时仍可能需要本地 tokenizer / label_map（除非 LoadedModel 自带）
        bundled = self._injected if isinstance(self._injected, LoadedModel) else None
        need_aux = bundled is None or bundled.tokenizer is None or bundled.label_map is None
        if need_aux:
            aux = [f for f in self._repo.required_files(self.precision)
                   if not f.endswith(".onnx")]
            missing = [f for f in aux if not (self.model_dir / f).exists()]
            if missing and self.auto_download:
                AsyncModelDownloader.download_sync(
                    missing, self.model_dir, self._repo,
                    optional=self._repo.optional_files(),
                )
            elif missing:
                # 只有 label_map 是硬依赖（除非调用方注入了 label_map）
                hard = [f for f in missing if f == "label_map.json"]
                if hard and self._injected_label_map is None:
                    raise FileNotFoundError(
                        f"缺少标签映射: {hard}；请设置 auto_download=True "
                        f"或传入 label_map=..."
                    )
                logger.warning("[初始化] 缺少可选辅助文件: %s", missing)
        loaded = attach_model(
            self._injected,
            precision=self._precision,
            backend=self.backend if self.backend in ("rust", "onnx") else "onnx",
            owns=self._owns_model,
            replace=self._replace,
            tokenizer=self._injected_tokenizer,
            label_map=self._injected_label_map,
            calib=self._injected_calib,
            num_threads=self._num_threads,
        )
        self._model_source = f"external/{loaded.source}"
        self.__adopt(loaded)

    def __init_backend(self):
        """后端链：Rust(ort 整图) → Python ONNX → PyTorch；全程单一实例。"""
        assert self._repo is not None
        if self.backend == "auto":
            backends = ("rust", "onnx")
        elif self.backend in ("rust", "onnx"):
            backends = (self.backend,)
        else:
            backends = ("pytorch",)
        # 只读预检：配置冲突必须直接报错，不能被后端回退链吞掉后
        # 继续尝试下一个后端、甚至绕过单例再加载一份 PyTorch 权重。
        assert_compatible(
            model_dir=self.model_dir, precision=self._precision,
            num_threads=self._num_threads, max_batch=self._max_batch,
            max_seq_len=self.max_seq_len, use_lang_token=self.use_lang_token,
            replace=self._replace, backends=backends,
        )
        shared = current_model()
        if shared is not None and self._replace:
            # 旧实现在这里直接复用共享实例，replace=True 形同虚设
            release_model()
            shared = None
        if shared is not None:
            # 已有实例（可能是另一个分类器创建的）：复用，不再初始化
            logger.info("[初始化] 复用进程内唯一实例（%s/%s）",
                        shared.backend, shared.precision)
            self._model_source = f"shared/{shared.source}"
            self.__adopt(shared)
            return

        if self.backend in ("auto", "rust"):
            try:
                loaded = get_engine(
                    model_dir=self.model_dir,
                    tokenizer_dir=self.tokenizer_dir if self.tokenizer_dir else None,
                    precision=self._precision,
                    backend="rust", num_threads=self._num_threads,
                    max_batch=self._max_batch, max_seq_len=self.max_seq_len,
                    use_lang_token=self.use_lang_token, replace=self._replace,
                )
                self._model_source = "internal/rust-ort"
                self.__adopt(loaded)
                logger.info("Rust 内核就绪（ort 整图 / %s）", self._precision)
                return
            except (ModelLoadError, BackendError) as exc:
                if self.backend == "rust":
                    raise
                logger.info("Rust 内核不可用（%s），回退到 ONNX Runtime", exc)

        if self.backend in ("auto", "onnx"):
            try:
                loaded = get_engine(
                    model_dir=self.model_dir,
                    tokenizer_dir=self.tokenizer_dir if self.tokenizer_dir else None,
                    precision=self._precision,
                    backend="onnx",
                    num_threads=self._num_threads,
                    max_batch=self._max_batch,
                    max_seq_len=self.max_seq_len,
                    use_lang_token=self.use_lang_token,
                    replace=self._replace,
                )
                self._model_source = "internal/python-ort"
                self.__adopt(loaded)
                logger.info("ONNX Runtime 后端就绪（%s）", self._precision)
                return
            except (ModelLoadError, BackendError) as exc:
                if self.backend == "onnx":
                    raise
                logger.info("ONNX 后端不可用（%s），回退到 PyTorch", exc)

        self.__init_pytorch()

    def __adopt(self, loaded: LoadedModel):
        """把 LoadedModel 绑定到本实例（不复制、不重新加载）。"""
        assert self._repo is not None
        self._model = loaded
        self._precision = loaded.precision
        if loaded.backend == "rust":
            self._rust_engine = loaded.session
            self.backend = "rust"
            self._whole_graph = True
            return
        if loaded.backend == "onnx":
            self._session = loaded.session
            self.backend = "onnx"
            self._input_names = [i.name for i in self._session.get_inputs()]
            self._output_names = [o.name for o in self._session.get_outputs()]
            self._whole_graph = self._repo.is_whole_graph(self._session)
            if self._whole_graph:
                self.__init_tokenizer(loaded.tokenizer)
            else:
                self.__init_encoder()
            return
        if loaded.backend == "pytorch":
            # 旧两段式产物：共享分类头，编码器仍是实例级资源
            self._pt_model = loaded.session
            self.backend = "pytorch"
            self._whole_graph = False
            if self._encoder is None:
                self.__init_encoder()
            return
        raise BackendError(f"无法接管后端: {loaded.backend}")

    # ── tokenizer / 标签 / 校准 ──
    def __init_tokenizer(self, injected: Any = None):
        """部署目录自带 tokenizer，绝不在推理时联网拉 hub。"""
        if injected is not None:
            if hasattr(injected, "__call__") and hasattr(injected, "get_vocab"):
                self._tokenizer = injected
            else:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(injected),
                    local_files_only=True
                )
        else:
            from transformers import AutoTokenizer
            local = self._repo.find_tokenizer() if self._repo else None
            env_dir = os.environ.get("EMOTION_TOKENIZER_DIR")
            candidates = [
                local,
                Path(env_dir) if env_dir else None,
                Path(self.tokenizer_dir)
            ]
            for cand in candidates:
                if cand is not None and Path(cand).is_dir():
                    try:
                        self._tokenizer = AutoTokenizer.from_pretrained(
                            str(cand), local_files_only=True)
                        break
                    except Exception:
                        continue
            if self._tokenizer is None:
                raise FileNotFoundError(
                    f"部署目录缺少 tokenizer: {self.model_dir}（需要 tokenizer/ 或 "
                    f"sentencepiece.bpe.model）\n"
                    f"请设置 auto_download=True 从 Release 获取 tokenizer.zip，"
                    f"或先跑 scripts/export/package_deploy.py"
                )
        # ★ v0.1 死代码修复：这段检查原本写在 raise 之后，永远不会执行
        self.__check_lang_tokens()

    def __check_lang_tokens(self):
        if not self.use_lang_token or self._tokenizer is None:
            return
        try:
            vocab = self._tokenizer.get_vocab()
        except Exception:
            return
        missing = [t for t in ModelConfig.LANG_TOKENS.values() if t not in vocab]
        if missing:
            logger.warning("词表缺少语言标记 %s，本次运行不加 lang token", missing)
            self.use_lang_token = False

    def __load_label_map(self, mapping: Any = None):
        if self._label2id:
            return
        if mapping is None:
            lm_path = ModelConfig.get_label_map_path(self.model_dir)
            if not lm_path.exists():
                raise FileNotFoundError(f"标签映射文件不存在: {lm_path}")
            with open(lm_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        self._set_label_map(mapping)

    def _set_label_map(self, mapping: Any):
        if isinstance(mapping, (list, tuple)):
            self._id2label = {i: str(v) for i, v in enumerate(mapping)}
            self._label2id = {str(v): i for i, v in enumerate(mapping)}
        elif isinstance(mapping, dict) and ("id2label" in mapping or "label2id" in mapping):
            self._id2label = {int(k): str(v) for k, v in
                              (mapping.get("id2label") or {}).items()}
            self._label2id = dict(mapping.get("label2id") or
                                  {v: k for k, v in self._id2label.items()})
        elif isinstance(mapping, dict):
            # {"高兴": 0, ...} 或 {"0": "高兴", ...}
            if all(isinstance(v, int) for v in mapping.values()):
                self._label2id = {str(k): int(v) for k, v in mapping.items()}
                self._id2label = {v: k for k, v in self._label2id.items()}
            else:
                self._id2label = {int(k): str(v) for k, v in mapping.items()}
                self._label2id = {v: k for k, v in self._id2label.items()}
        else:
            raise InvalidInputError(f"无法解析 label_map: {type(mapping)}")
        self._num_labels = max(self._id2label.keys(), default=-1) + 1

    @staticmethod
    def __coerce_calib(value: Any) -> Calibration:
        if value is None:
            return Calibration()
        if isinstance(value, Calibration):
            return value
        if isinstance(value, dict):
            return Calibration(
                temperature=float(value.get("temperature", 1.0)),
                per_class=value.get("per_class") or {},
                per_lang=value.get("per_lang") or {},
            )
        return Calibration.load(value)

    def __init_encoder(self):
        """非整图旧产物：单独编码器 ONNX + 分类头。"""
        if self.encoder_path:
            from .models.encoder import ONNXEncoder
            self._encoder = ONNXEncoder(
                str(self.encoder_path),
                tokenizer_dir=self.tokenizer_dir,
                cache_size=self._encoder_cache_size,
                strict_dim=False
            )
            return
        enc_onnx = self._repo.find_encoder_onnx() if self._repo else None
        if enc_onnx:
            from .models.encoder import ONNXEncoder
            self._encoder = ONNXEncoder(
                str(enc_onnx),
                tokenizer_dir=self.tokenizer_dir,
                cache_size=self._encoder_cache_size,
                strict_dim=False
            )
            return
        logger.info("未找到 ONNX 编码器，回退到 SentenceTransformer（v1 兼容）...")
        from .models.encoder import SentenceTransformerEncoder
        self._encoder = SentenceTransformerEncoder(device=self.device)

    def __init_pytorch(self):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch 后端需要安装 torch。pip install torch") from exc
        from .models.pytorch import EmotionClassifier as PTModel

        pt_path = self._repo.find_pytorch_weight() if self._repo else None
        if pt_path is None:
            raise FileNotFoundError(
                f"未找到 PyTorch 权重: {self.model_dir / 'emotion_classify.pt'}"
            )
        ckpt = torch.load(pt_path, map_location=self.device)
        input_dim = int(ckpt.get("input_dim", ModelConfig.EXPECTED_DIM))
        hidden_dim = int(ckpt.get("hidden_dim", 128))
        self._pt_model = PTModel(
            input_dim=input_dim,
            num_classes=self._num_labels or 18,
            hidden_dim=hidden_dim,
            dropout=0.0
        ).to(self.device)
        self._pt_model.load_state_dict(ckpt.get("state_dict", ckpt))
        self._pt_model.eval()
        if self._encoder is None:
            self.__init_encoder()
        self.backend = "pytorch"
        self._whole_graph = False
        # 旧两段式 PyTorch 产物同样纳入唯一实例约束：否则回退链会绕过
        # runtime，第二个分类器再加载一份权重。
        attach_model(
            LoadedModel(
                session=self._pt_model,
                precision="fp32",
                backend="pytorch",
                owns=True,
                source="internal",
                meta={"model_dir": str(self.model_dir)}
            ),
            precision="fp32", backend="pytorch", owns=True,
            replace=self._replace,
        )
        logger.info("PyTorch 后端就绪: %s", pt_path.name)

    # ── 推理 ──
    def encode(
        self,
        texts: Union[str, List[str]], batch_size: int = 32,
        langs: Optional[List[Optional[str]]] = None
    ) -> np.ndarray:
        """返回整图 logits（或旧两段式产物的句向量）。"""
        self.ensure_loaded()
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self._num_labels), dtype=np.float32)
        if self._rust_engine is not None:
            if langs is None:
                cleaned = None                       # 交给 Rust 侧统一检测
            else:
                if len(langs) != len(texts):
                    raise InvalidInputError("lang 数量与文本数量不一致")
                # 部分指定时逐条补齐：旧实现只要有一个 None 就把整批退回
                # 自动检测，用户显式给的语言前缀被静默丢掉。
                cleaned = [str(l) if l is not None else detect_lang(t or "")
                           for t, l in zip(texts, langs)]
            return np.asarray(
                self._rust_engine.forward(list(texts), cleaned), dtype=np.float32)
        if self._whole_graph:
            return self.__onnx_logits(texts, langs, batch_size)
        if self._pt_model is not None:
            import torch
            embs = self._encoder.encode( # type: ignore
                texts,
                batch_size=batch_size,
                normalize=False
            )
            with torch.no_grad():
                logits = self._pt_model(
                    torch.from_numpy(np.asarray(embs, dtype=np.float32))
                    .to(self.device))[0]
            return logits.detach().cpu().numpy().astype(np.float32)
        return self._encoder.encode(texts, batch_size=batch_size)  # type: ignore

    def __onnx_logits(
        self, texts: List[str],
        langs: Optional[List[Optional[str]]],
        batch_size: int = 32
    ) -> np.ndarray:
        assert self._tokenizer is not None and self._session is not None
        if langs is None and self.use_lang_token:
            langs = detect_langs(texts) # type: ignore
        max_len = self.max_seq_len
        output = "logits" if "logits" in self._output_names else self._output_names[0]
        outs = []
        for s in range(0, len(texts), max(1, batch_size)):
            chunk = list(texts[s:s + batch_size])
            chunk_langs = list(langs[s:s + batch_size]) if langs else [None] * len(chunk)
            prepared = [apply_lang_token(t, l, self.use_lang_token)
                        for t, l in zip(chunk, chunk_langs)]
            enc = self._tokenizer(prepared, padding=True, truncation=True,
                                  max_length=max_len, return_tensors="np")
            feeds: Dict[str, np.ndarray] = {}
            for name in self._input_names:
                if name in enc:
                    feeds[name] = np.asarray(enc[name], dtype=np.int64)
                elif name == "token_type_ids":
                    feeds[name] = np.zeros_like(
                        np.asarray(enc["input_ids"]), dtype=np.int64)
                else:
                    raise InvalidInputError(f"ONNX 模型要求未知输入 '{name}'")
            outs.append(
                np.asarray(self._session.run([output], feeds)[0],
                dtype=np.float32)
            )
        return np.concatenate(outs, axis=0)

    def predict_proba(self, texts: Union[str, List[str]], batch_size: int = 32,
                      lang: Optional[Union[str, List[Optional[str]]]] = None) -> np.ndarray:
        """返回 (N, num_labels) 的概率矩阵。"""
        self.ensure_loaded()
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if isinstance(t, str) else "" for t in texts]
        if not texts:
            return np.zeros((0, self._num_labels), dtype=np.float32)

        langs = normalize_langs(lang if isinstance(lang, list) else
                                ([lang] if lang else None), len(texts))
        if langs is None and self.use_lang_token:
            langs = detect_langs(texts)
        if langs is not None and any(l is None for l in langs):
            # 部分指定语言：补齐剩余项，避免 ONNX 路径不加前缀、
            # Rust 路径因收到半截 None 而整批退回自动检测
            langs = [l if l is not None else detect_lang(t or "")
                     for t, l in zip(texts, langs)]

        logits = np.array(
            self.encode(texts, batch_size, langs=langs), # type: ignore
            dtype=np.float32,
            copy=True
        )

        # 温度校准（per-language 可选）
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
        lang: Optional[Union[str, List[Optional[str]]]] = None,
    ) -> List[List[Tuple[str, float]]]:
        """情感分类预测，返回 [[(label, score), ...], ...]。"""
        return [
            self._to_pairs(p, top_k)
            for p in self.predict_proba(texts, batch_size=batch_size, lang=lang)
        ]

    def predict_label(
        self,
        texts: Union[str, List[str]],
        lang: Optional[Union[str, List[Optional[str]]]] = None
    ) -> List[List[Tuple[str, float]]]:
        """只返回最可能的 1 个标签。"""
        return self.predict(texts, top_k=1, lang=lang)

    def _to_pairs(self, prob_vec: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        k = max(1, min(int(top_k), len(prob_vec)))
        top_idx = np.argpartition(-prob_vec, k - 1)[:k]
        top_idx = top_idx[np.argsort(-prob_vec[top_idx])]
        return [
            (self._id2label.get(int(i), str(i)),
                 float(round(float(prob_vec[i]), 4))) for i in top_idx
        ]

    def predict_batch(
        self,
        texts: List[str],
        top_k: int = 3,
        batch_size: int = 64
    ) -> List[List[Tuple[str, float]]]:
        return self.predict(texts, top_k=top_k, batch_size=batch_size)

    def get_labels(self) -> List[str]:
        if not self._label2id:
            with self._load_lock:
                if not self._label2id:
                    self.__load_label_map(self._injected_label_map)
        return list(self._label2id.keys())

    # ── 运维 ──
    def list_available_models(self) -> Dict[str, List[str]]:
        import glob
        result = {"classifier_onnx": [], "encoder_onnx": [], "pytorch": [], "calib": []}
        for f in sorted(glob.glob(str(self.model_dir / "emotion_classifier*.onnx"))):
            result["classifier_onnx"].append(Path(f).name)
        for f in glob.glob(str(self.model_dir / "model_*.onnx")) + \
                glob.glob(str(self.model_dir / "xlmr_*.onnx")):
            result["encoder_onnx"].append(Path(f).name)
        if (self.model_dir / "emotion_classify.pt").exists():
            result["pytorch"].append("emotion_classify.pt")
        if (self.model_dir / CALIB_NAME).exists():
            result["calib"].append(CALIB_NAME)
        return result

    def update_models(self):
        """强制刷新 Release 资产：先释放单例，避免旧 mmap/句柄指向被覆盖文件。"""
        self.close()
        release_model()
        repo = self._repo or ModelRepository(self.model_dir, repo_url=self.repo_url)
        self._repo = repo
        needed = repo.required_files(self.precision)
        for f in needed:
            fp = self.model_dir / f
            if fp.exists() and fp.is_file():
                fp.unlink()
        AsyncModelDownloader.download_sync(needed, self.model_dir, repo,
                                          optional=repo.optional_files())
        self._loaded = False
        logger.info("模型更新完成，下次预测将重新初始化唯一实例")

    def model_info(self) -> Dict[str, Any]:
        if not self._loaded:
            with self._load_lock:
                if not self._loaded:
                    self.__load_label_map(self._injected_label_map)
        repo = self._repo or ModelRepository(self.model_dir, repo_url=self.repo_url)
        shared = current_model()
        info: Dict[str, Any] = {
            "version": ModelConfig.VERSION,
            "backend": self.backend,
            "precision": self._precision,
            "model_source": self._model_source,
            "singleton": shared is not None,
            "singleton_backend": shared.backend if shared else None,
            "encoder": ModelConfig.ENCODER_NAME,
            "whole_graph": self._whole_graph,
            "model_dir": str(self.model_dir),
            "max_seq_len": self.max_seq_len,
            "use_lang_token": self.use_lang_token,
            "num_labels": self._num_labels,
            "labels": self.get_labels(),
            "device": self.device,
            "calibrated": self._calib is not None
            and abs(self._calib.temperature - 1.0) > 1e-6,
        }
        info["files"] = {}
        for name in (
            *MODEL_FILES.values(),
            "emotion_classify.pt", "label_map.json",
            CALIB_NAME, 
            MANIFEST_NAME
        ):
            fp = self.model_dir / name
            if name == "label_map.json":
                #先从项目/data/label_map.json读取，如果不存在再从release下载
                lm_path = Path(__file__).parent.parent / "data" / "label_map.json"
                if lm_path.exists():
                    fp = lm_path
            info["files"][name] = {
                "exists": fp.exists(),
                "size_mb": round(fp.stat().st_size / 1e6, 2) if fp.is_file() else 0,
            }
        cls_path = repo.find_classifier_onnx(self.precision)
        info["classifier"] = cls_path.name if cls_path else None
        info["release"] = repo.repo_url
        return info

    @staticmethod
    def release_shared_model() -> bool:
        """显式释放进程内唯一实例（程序管理模型生命周期的出口）。"""
        return release_model()

    def close(self):
        """只解除本对象引用；共享单例由程序通过 release_shared_model() 释放。"""
        try:
            if getattr(self, "_encoder", None) is not None and \
                    hasattr(self._encoder, "close"):
                self._encoder.close()  # type: ignore
        except Exception:
            pass
        self._encoder = None
        self._session = None
        self._pt_model = None
        self._rust_engine = None
        self._tokenizer = None
        self._model = None
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
