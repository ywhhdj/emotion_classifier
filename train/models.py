import os
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


class StochasticDepth(nn.Module):
    """随机深度：训练时以概率 p 丢弃整个残差分支。"""

    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep_prob = 1 - self.p
        mask = torch.empty(x.size(0), 1, device=x.device).bernoulli_(keep_prob)
        return x / keep_prob * mask.to(x.dtype)


class LightEmotionClassifier(nn.Module):
    """轻量情感分类头：投影层 + 单层残差块 + 线性分类。"""

    def __init__(
        self,
        input_dim=384,
        num_classes=19,
        hidden_dim=128,
        dropout=0.3,
        stochastic_p=0.2
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stochastic_depth = StochasticDepth(p=stochastic_p)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, label=None):
        x = self.proj(x)
        residual = self.stochastic_depth(self.residual(x))
        x = x + residual
        logits = self.classifier(x)
        return logits, x, None


class FocalLoss(nn.Module):
    """Focal Loss。

    解决类别不平衡，聚焦难分类样本。
        loss = -(1-pt)^gamma * log(pt)
    """

    def __init__(self, gamma=2.0, weight=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1).exp()  # 未加权的真实概率
        ce = F.nll_loss(log_probs, targets, weight=self.weight,
                        reduction='none', ignore_index=-100)
        if self.weight is not None:
            # nll_loss 的 weight 会缩放 loss，这里还原成「纯 CE × 类别权重」
            w = self.weight.gather(0, targets).clamp_min(1e-6)
            ce = ce / w * w.detach()
        loss = ((1 - pt) ** self.gamma) * ce

        if self.label_smoothing > 0:
            smooth = -log_probs.mean(dim=-1) * self.label_smoothing
            loss = loss * (1 - self.label_smoothing) + smooth

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class ONNXEncoder:

    def __init__(
        self,
        onnx_path: str,
        tokenizer_dir: Optional[str] = None,
        tokenizer_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_tokens: int = 120,
        overlap: int = 20,
    ):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.max_tokens = max_tokens
        self.overlap = overlap
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = max(1, min(8, os.cpu_count() or 4))
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs_info = {i.name: i for i in self.sess.get_inputs()}
        self.output_name = self.sess.get_outputs()[0].name

        if tokenizer_dir and os.path.isdir(tokenizer_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _chunk_ids(self, text: str) -> List[List[int]]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        limit = max(1, self.max_tokens - 2)
        if len(ids) <= limit:
            return [ids]
        step = max(1, limit - self.overlap)
        chunks = [ids[i:i + limit] for i in range(0, len(ids), step)]
        if len(chunks) > 1 and len(chunks[-1]) < max(1, limit // 4):
            chunks[-2].extend(chunks[-1])
            chunks.pop()
        return chunks

    def _pack(self, id_chunks: Sequence[Sequence[int]]) -> dict:
        cls_id = self.tokenizer.cls_token_id or 0
        sep_id = self.tokenizer.sep_token_id or 0
        pad_id = self.tokenizer.pad_token_id or 0
        seqs = [[cls_id] + list(c) + [sep_id] for c in id_chunks]
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

    def _forward(self, enc: dict) -> np.ndarray:
        feeds = {}
        for name in self.inputs_info:
            if name in enc:
                feeds[name] = enc[name].astype(np.int64)
            elif name == "token_type_ids":
                feeds[name] = np.zeros_like(enc["input_ids"])
            else:
                raise ValueError(f"ONNX 模型要求未知输入 '{name}'")
        out = self.sess.run([self.output_name], feeds)[0]
        if out.ndim == 3: # type: ignore
            mask = enc["attention_mask"][:, :, None].astype(np.float32)
            out = (out * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        return out.astype(np.float32) # type: ignore

    def encode(self, texts, batch_size=32, normalize=True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if isinstance(t, str) else "" for t in texts]
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        flat: List[List[int]] = []
        owner: List[int] = []
        spans: List[tuple] = []
        for i, t in enumerate(texts):
            chunks = self._chunk_ids(t)
            spans.append((i, len(chunks)))
            for c in chunks:
                flat.append(c)
                owner.append(i)

        # 按长度排序分桶：padding 更少、batch 内 shape 更整齐
        order = sorted(range(len(flat)), key=lambda k: len(flat[k]))
        dim = None
        vecs_by_idx: List[np.ndarray] = [None] * len(flat)  # type: ignore
        for s in range(0, len(order), batch_size):
            idxs = order[s:s + batch_size]
            res = self._forward(self._pack([flat[k] for k in idxs]))
            dim = res.shape[-1]
            for j, k in enumerate(idxs):
                vecs_by_idx[k] = res[j]
        assert dim is not None

        out = np.zeros((len(texts), dim), dtype=np.float32)
        cursor = 0
        for i, n_chunks in spans:
            out[i] = np.mean(vecs_by_idx[cursor:cursor + n_chunks], axis=0)
            cursor += n_chunks

        if normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.clip(norms, 1e-9, None)
        return out.astype(np.float32)


class PoolingStrategy:
    MEAN = "mean"
    MAX = "max"
    WEIGHTED = "weighted"


class ChunkedEncoder:
    """超长文本分块编码器（SentenceTransformer 后端）"""

    def __init__(self, st_model, max_tokens=120, overlap=20, strategy=PoolingStrategy.MEAN):
        self.st = st_model
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.strategy = strategy
        self.tokenizer = getattr(st_model, "tokenizer", None)
        if self.tokenizer is None:
            first = getattr(st_model, "_first_module", None)
            if callable(first):
                try:
                    self.tokenizer = getattr(first(), "tokenizer", None)
                except Exception:
                    self.tokenizer = None

    def _chunk(self, text: str) -> List[str]:
        step = max(1, self.max_tokens - self.overlap)
        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) <= self.max_tokens:
                return [text]
            chunks = [tokens[i:i + self.max_tokens] for i in range(0, len(tokens), step)]
            # 中文 WordPiece 会在字间插空格，去掉多余空格保持原文连续
            return [c.replace(" ", "") if not (" " in text and "".join(c.split())) else "".join(c)
                    for c in chunks]
        # 无 tokenizer：中文按字符切，纯英文按空格切
        units = text.split() if (" " in text and not _has_cjk(text)) else list(text)
        if len(units) <= self.max_tokens:
            return [text]
        sep = " " if units is not text and " " in text else ""
        return [sep.join(units[i:i + self.max_tokens]) for i in range(0, len(units), step)]

    def encode(self, text: str) -> np.ndarray:
        chunks = self._chunk(text)
        if not chunks:
            return np.zeros(self.st.get_sentence_embedding_dimension(), dtype=np.float32)

        vecs = self.st.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)

        if self.strategy == PoolingStrategy.MEAN:
            return vecs.mean(axis=0)
        if self.strategy == PoolingStrategy.MAX:
            return vecs.max(axis=0)
        if self.strategy == PoolingStrategy.WEIGHTED:
            weights = np.array(
                [1.0 + 0.3 * ("!" in c or "！" in c) + 0.2 * ("?" in c or "？" in c) for c in chunks],
                dtype=np.float32,
            )
            weights = weights / weights.sum()
            return (weights[:, None] * vecs).sum(axis=0)
        raise ValueError(f"未知池化策略: {self.strategy}")


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


class EmotionDataset(Dataset):
    """可选的惰性数据集：已有 embedding 时直接取向量，否则返回原文。"""

    def __init__(self, texts, labels, embeddings=None):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.embeddings = embeddings

    def set_embeddings(self, emb: np.ndarray):
        self.embeddings = emb.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.embeddings is not None:
            return self.embeddings[idx], self.labels[idx]
        return self.texts[idx], self.labels[idx]