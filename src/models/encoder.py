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
    # 推理进程通常只需少量线程；过多线程在小 batch 下反而因同步变慢
    intra = max(1, min(8, cpu))
    opts.intra_op_num_threads = int(os.environ.get("EMO_ORT_INTRA", intra))
    opts.inter_op_num_threads = int(os.environ.get("EMO_ORT_INTER", 1))
    return opts


class ONNXEncoder(BaseEncoder):
    """
    批量 ONNX 句向量编码器。

    修复要点（原实现逐条 encode，batch_size 形同虚设）：
      1. 真正的批推理：短文本按长度分桶后整批前向，长文本先切块再整批前向；
      2. token 级切块直接用 input_ids，避免 convert_tokens_to_string 把中文拆出空格；
      3. 维度校验优先用静态 shape，省掉一次冷启动 dummy 前向；
      4. 可选 LRU 文本缓存，交互/重复语料场景直接命中。
    """
    EXPECTED_DIM = 384

    def __init__(
        self,
        onnx_path: str,
        tokenizer_dir: Optional[str] = None,
        tokenizer_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_tokens: int = 120,
        overlap: int = 20,
        cache_size: int = 0,
        providers: Optional[Sequence[str]] = None,
    ):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if max_tokens <= 2:
            raise ValueError("max_tokens 必须大于 2（需预留 [CLS]/[SEP]）")
        if not (0 <= overlap < max_tokens - 2):
            raise ValueError("overlap 需满足 0 <= overlap < max_tokens-2")

        self.max_tokens = max_tokens
        self.overlap = overlap
        self._providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(
            onnx_path, sess_options=_default_session_options(), providers=self._providers
        )
        self.inputs_info = {i.name: i for i in self.sess.get_inputs()}
        self.outputs_info = {o.name: o for o in self.sess.get_outputs()}
        self.output_name = self.sess.get_outputs()[0].name

        # 加载 tokenizer（本地目录优先，避免每次联网）
        if tokenizer_dir and os.path.isdir(tokenizer_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self._dim = self.__infer_dim()
        if self._dim is not None and self._dim != self.EXPECTED_DIM:
            raise ValueError(
                f"编码器输出维度应为 {self.EXPECTED_DIM}，实际为 {self._dim}。"
                f"请提供输出维度为 {self.EXPECTED_DIM} 的 ONNX 编码器。"
            )

        self._cache: "dict[str, np.ndarray]" = {}
        self._cache_size = int(cache_size)
        self._lock = threading.Lock()

    # ── 维度推断：优先静态 shape，失败再跑一次 dummy ──
    def __infer_dim(self) -> Optional[int]:
        shape = self.outputs_info[self.output_name].shape
        if shape and isinstance(shape[-1], int) and shape[-1] > 0:
            return int(shape[-1])
        try:
            out = self.__forward(self.__dummy_batch(1))
        except Exception:
            return None
        return int(out.shape[-1])

    # ── 前向 ──
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
        out = self.sess.run([self.output_name], self.__build_feeds(enc))[0] #type: ignore
        if out.ndim == 3:  #type: ignore
            mask = enc["attention_mask"][:, :, None].astype(np.float32)
            out = (out * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        return out.astype(np.float32) #type: ignore

    # ── 分块 ──
    def __chunk_ids(self, text: str) -> List[List[int]]:
        """按 token 切块（保留 max_tokens-2 的有效 token，给 [CLS]/[SEP] 留位）。"""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        limit = self.max_tokens - 2
        if len(ids) <= limit:
            return [ids]
        step = max(1, limit - self.overlap)
        chunks = [ids[i:i + limit] for i in range(0, len(ids), step)]
        # 防止尾部产生极短碎片块
        if len(chunks) > 1 and len(chunks[-1]) < limit // 4:
            chunks[-2].extend(chunks[-1])
            chunks.pop()
        return chunks

    def _pack(self, id_chunks: List[List[int]]) -> dict:
        cls_id = self.tokenizer.cls_token_id or 0
        sep_id = self.tokenizer.sep_token_id or 0
        pad_id = self.tokenizer.pad_token_id or 0
        seqs = [[cls_id] + c + [sep_id] for c in id_chunks]
        max_len = max(len(s) for s in seqs)
        input_ids = np.full((len(seqs), max_len), pad_id, dtype=np.int64)
        attention = np.zeros((len(seqs), max_len), dtype=np.int64)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = s
            attention[i, :len(s)] = 1
        enc = {"input_ids": input_ids, "attention_mask": attention}
        if "token_type_ids" in self.inputs_info:
            enc["token_type_ids"] = np.zeros_like(input_ids)
        return enc


    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if isinstance(t, str) else "" for t in texts]
        if not texts:
            return np.zeros((0, self.EXPECTED_DIM), dtype=np.float32)

        out = np.zeros((len(texts), self._dim or self.EXPECTED_DIM), dtype=np.float32)
        todo: List[int] = []

        # 缓存命中
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
            # 先切块，得到 (样本下标 → 块列表)
            plan: List[tuple] = []
            flat: List[List[int]] = []
            owner: List[int] = []
            for i in todo:
                chunks = self.__chunk_ids(texts[i])
                plan.append((i, len(chunks)))
                for c in chunks:
                    flat.append(c)
                    owner.append(i)

            # 按长度排序分桶 → padding 浪费最小、batch 内 shape 统一
            order = sorted(range(len(flat)), key=lambda k: len(flat[k]))
            vecs = np.zeros((len(flat), out.shape[1]), dtype=np.float32)
            for s in range(0, len(order), batch_size):
                idxs = order[s:s + batch_size]
                enc = self._pack([flat[k] for k in idxs])
                res = self.__forward(enc)
                for j, k in enumerate(idxs):
                    vecs[k] = res[j]

            # 按样本聚合（多块 mean pooling）
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


class SentenceTransformerEncoder(BaseEncoder):
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self.model is None:
            raise ValueError("SentenceTransformer 编码器已关闭")
        embs = self.model.encode(
            texts,
            batch_size=batch_size or self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=normalize,
        )
        return np.asarray(embs, dtype=np.float32)

    def close(self):
        self.model = None