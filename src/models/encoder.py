"""编码器抽象（v2.0：XLM-R 768 维 + Rust 内核优先）。

后端优先级：rust（内部内核，无 GIL） > onnx（回退） > pytorch。
对外接口 encode(texts, batch_size, normalize) -> np.ndarray 保持不变。
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Sequence
import os
import threading

import numpy as np


class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        pass

    def close(self):
        pass


def _default_session_options():
    """统一的 ONNX Runtime 会话配置：开满图优化 + 限制线程避免过订阅。"""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    cpu = os.cpu_count() or 4
    intra = max(1, min(8, cpu))
    opts.intra_op_num_threads = int(os.environ.get("EMO_ORT_INTRA", intra))
    opts.inter_op_num_threads = int(os.environ.get("EMO_ORT_INTER", 1))
    return opts


class ONNXEncoder(BaseEncoder):
    """XLM-R ONNX 编码器（回退路径）。

    v2 变化：
      * 默认 tokenizer 改为 xlm-roberta-base，维度 768（v1 为 MiniLM 384）；
      * XLM-R 的 pad_id 是 1（不是 0）；
      * 不再需要 token_type_ids。
    """

    EXPECTED_DIM = 768

    def __init__(
        self,
        onnx_path: str,
        tokenizer_dir: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        max_tokens: int = 48,
        overlap: int = 20,
        cache_size: int = 0,
        providers: Optional[Sequence[str]] = None,
        strict_dim: bool = False,
    ):
        import onnxruntime as ort

        tokenizer_name = tokenizer_name or "xlm-roberta-base"
        if max_tokens <= 2:
            raise ValueError("max_tokens 必须大于 2（需预留 <s>/</s>）")

        self.max_tokens = max_tokens
        self.overlap = overlap
        self._providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(
            onnx_path, sess_options=_default_session_options(), providers=self._providers
        )
        self.inputs_info = {i.name: i for i in self.sess.get_inputs()}
        self.outputs_info = {o.name: o for o in self.sess.get_outputs()}
        self.output_name = self.sess.get_outputs()[0].name

        from transformers import AutoTokenizer
        if tokenizer_dir and os.path.isdir(tokenizer_dir) and any(
                f.startswith("tokenizer") or f.startswith("sentencepiece")
                for f in os.listdir(tokenizer_dir)):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1

        self._dim = self.__infer_dim()
        if strict_dim and self._dim is not None and self._dim != self.EXPECTED_DIM:
            raise ValueError(f"编码器输出维度应为 {self.EXPECTED_DIM}，实际为 {self._dim}")

        self._cache: "dict[str, np.ndarray]" = {}
        self._cache_size = int(cache_size)
        self._lock = threading.Lock()

    def __infer_dim(self) -> Optional[int]:
        shape = self.outputs_info[self.output_name].shape
        if shape and isinstance(shape[-1], int) and shape[-1] > 0:
            return int(shape[-1])
        try:
            out = self.__forward(self.__dummy_batch(1))
        except Exception:
            return None
        return int(out.shape[-1])

    def __dummy_batch(self, n: int) -> dict:
        return self.tokenizer(["test"] * n, padding=True, truncation=True,
                              max_length=16, return_tensors="np")

    def __build_feeds(self, enc: dict) -> dict:
        feeds = {}
        for input_name in self.inputs_info:
            if input_name in ("input_ids", "attention_mask", "token_type_ids"):
                arr = enc.get(input_name)
                if arr is None:
                    arr = np.zeros_like(enc["input_ids"])
                feeds[input_name] = arr.astype(np.int64)
            elif input_name in enc:
                feeds[input_name] = enc[input_name].astype(np.int64)
            else:
                raise ValueError(f"ONNX 模型要求未知输入 '{input_name}'")
        return feeds

    def __forward(self, enc: dict) -> np.ndarray:
        out = self.sess.run([self.output_name], self.__build_feeds(enc))[0]  # type: ignore
        if out.ndim == 3:  # type: ignore
            mask = enc["attention_mask"][:, :, None].astype(np.float32)
            out = (out * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        return out.astype(np.float32)  # type: ignore

    def __chunk_ids(self, text: str) -> List[List[int]]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        limit = self.max_tokens - 2
        if len(ids) <= limit:
            return [ids]
        step = max(1, limit - self.overlap)
        chunks = [ids[i:i + limit] for i in range(0, len(ids), step)]
        if len(chunks) > 1 and len(chunks[-1]) < limit // 4:
            chunks[-2].extend(chunks[-1])
            chunks.pop()
        return chunks

    def _pack(self, id_chunks: List[List[int]]) -> dict:
        cls_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else 0
        sep_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else 2
        seqs = [[cls_id] + c + [sep_id] for c in id_chunks]
        max_len = max(len(s) for s in seqs)
        input_ids = np.full((len(seqs), max_len), self.pad_id, dtype=np.int64)
        attention = np.zeros((len(seqs), max_len), dtype=np.int64)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = s
            attention[i, :len(s)] = 1
        enc = {"input_ids": input_ids, "attention_mask": attention}
        if "token_type_ids" in self.inputs_info:
            enc["token_type_ids"] = np.zeros_like(input_ids)
        return enc

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True,
               lang: Optional[List[str]] = None) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if isinstance(t, str) else "" for t in texts]
        if not texts:
            return np.zeros((0, self._dim or self.EXPECTED_DIM), dtype=np.float32)

        out = np.zeros((len(texts), self._dim or self.EXPECTED_DIM), dtype=np.float32)
        todo: List[int] = []

        if self._cache_size:
            with self._lock:
                for i, t in enumerate(texts):
                    hit = self._cache.get(t)
                    if hit is not None:
                        out[i] = hit
                    else:
                        todo.append(i)
        else:
            todo = list(range(len(texts)))

        if todo:
            plan: List[tuple] = []
            flat: List[List[int]] = []
            for i in todo:
                chunks = self.__chunk_ids(texts[i])
                plan.append((i, len(chunks)))
                for c in chunks:
                    flat.append(c)

            order = sorted(range(len(flat)), key=lambda k: len(flat[k]))
            vecs = np.zeros((len(flat), out.shape[1]), dtype=np.float32)
            for s in range(0, len(order), batch_size):
                idxs = order[s:s + batch_size]
                res = self.__forward(self._pack([flat[k] for k in idxs]))
                for j, k in enumerate(idxs):
                    vecs[k] = res[j]

            cursor = 0
            for i, n_chunks in plan:
                v = vecs[cursor:cursor + n_chunks].mean(axis=0)
                cursor += n_chunks
                out[i] = v
                if self._cache_size:
                    with self._lock:
                        if len(self._cache) >= self._cache_size:
                            self._cache.pop(next(iter(self._cache)))
                        self._cache[texts[i]] = v

        if normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.clip(norms, 1e-9, None)
        return out.astype(np.float32)

    def __call__(self, texts, batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        return self.encode(texts, batch_size=batch_size, normalize=normalize)

    def close(self):
        self.sess = None
        self._cache.clear()


class RustEngineEncoder(BaseEncoder):
    """Rust 内核编码器（内部实现分词 + 批调度 + 前向 + 后处理）。

    Python 侧只做参数校验与结果搬运，不参与任何逐 token 计算。
    内核调用在 py.allow_threads 内完成，会释放 GIL，多线程可真正并行。
    """

    EXPECTED_DIM = 768

    def __init__(self, model_dir: str, num_threads: int = 4, max_batch: int = 32,
                 max_wait_ms: int = 2, backend: str = "candle", precision: str = "fp16",
                 cache_size: int = 0):
        try:
            from . import _core as core          # type: ignore
        except ImportError as e:
            raise ImportError(
                "Rust 内核不可用（emotion_classifier._core 未编译）。"
                "请安装带内核的 wheel，或设置 EMOTION_BACKEND=onnx 回退到 ONNX Runtime。"
            ) from e

        self._core = core
        self.engine = core.EmotionEngine(
            model_dir=model_dir,
            num_threads=num_threads,
            max_batch=max_batch,
            max_wait_ms=max_wait_ms,
            backend=backend,
            precision=precision,
        )
        self._dim = int(self.engine.hidden_size())
        self._cache: "dict[str, np.ndarray]" = {}
        self._cache_size = int(cache_size)
        self._lock = threading.Lock()

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = False,
               lang: Optional[List[str]] = None) -> np.ndarray:
        """encode 在此返回 logits（整图模式下向量即 logits）。

        为保持 BaseEncoder 接口一致而保留该签名；normalize 参数被忽略
        （概率化统一在 predictor 里做）。
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.asarray(self.engine.forward(texts, lang), dtype=np.float32)

    def predict(self, texts: List[str], top_k: int = 3,
                lang: Optional[List[str]] = None) -> List[List[tuple]]:
        """一步到位：分词 -> 批调度 -> 前向 -> softmax/阈值 -> top-k。"""
        return self.engine.predict(texts, top_k, lang)

    def close(self):
        try:
            self.engine.close()
        except Exception:
            pass
        self._cache.clear()


class SentenceTransformerEncoder(BaseEncoder):
    """[DEPRECATED] MiniLM 句向量编码器，仅为兼容 v1 产物保留。"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 device: str = "cpu", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self.model is None:
            raise ValueError("SentenceTransformer 编码器已关闭")
        embs = self.model.encode(texts, batch_size=batch_size or self._batch_size,
                                 convert_to_numpy=True, show_progress_bar=False,
                                 normalize_embeddings=normalize)
        return np.asarray(embs, dtype=np.float32)

    def close(self):
        self.model = None
