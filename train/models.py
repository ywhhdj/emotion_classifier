import os
import re
from typing import Optional
from torch.utils.data import Dataset
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class StochasticDepth(nn.Module):
    """随机深度：训练时以概率 p 丢弃整个残差分支"""
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep_prob = 1 - self.p
        mask = torch.empty(x.size(0), 1, device=x.device).bernoulli_(keep_prob)
        return x / keep_prob * mask


class LightEmotionClassifier(nn.Module):
    """
    轻量情感分类头：
    - 单层投影 (384 → hidden_dim)
    - 单层残差块
    - 线性分类器
    参数量约为原模型的 1/8
    """
    def __init__(self, input_dim=384, num_classes=19, hidden_dim=128, dropout=0.3, stochastic_p=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.stochastic_depth = StochasticDepth(p=stochastic_p)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, label=None):
        x = self.proj(x)
        residual = self.residual(x)
        residual = self.stochastic_depth(residual)
        x = x + residual
        logits = self.classifier(x)
        feat = x
        return logits, feat, None  # 兼容原接口


class FocalLoss(nn.Module):
    """
    解决类别不平衡，聚焦难分类样本。
        loss = -(1-pt)^gamma * log(pt)
    """
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ONNXEncoder:

    def __init__(
        self,
        onnx_path: str,
        tokenizer_dir: Optional[str] = None,
        tokenizer_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        用 ONNX Runtime 加载量化后的 SentenceTransformer 编码器。
        Parameters:
            onnx_path: ONNX 模型文件路径
            tokenizer_dir: 模型使用的 tokenizer 文件夹路径
            tokenizer_name: 模型使用的 tokenizer 名称
        """
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inputs_info = {i.name: i for i in self.sess.get_inputs()}
        self.output_name = self.sess.get_outputs()[0].name

        if tokenizer_dir and os.path.isdir(tokenizer_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode(self, texts, batch_size=32, normalize=True):
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="np"
            )
            feeds = {}
            for input_name in self.inputs_info:
                if input_name == "input_ids":
                    feeds[input_name] = enc["input_ids"].astype(np.int64)
                elif input_name == "attention_mask":
                    feeds[input_name] = enc["attention_mask"].astype(np.int64)
                elif input_name == "token_type_ids":
                    tti = enc.get("token_type_ids")
                    if tti is None:
                        tti = np.zeros_like(enc["input_ids"])
                    feeds[input_name] = tti.astype(np.int64)
                else:
                    if input_name in enc:
                        feeds[input_name] = enc[input_name].astype(np.int64)
                    else:
                        raise ValueError(f"ONNX 模型要求未知输入 '{input_name}'")

            output = self.sess.run([self.output_name], feeds)[0]

            if output.ndim == 3:  # type: ignore
                mask = enc["attention_mask"][:, :, None].astype(np.float32)
                output = (output * mask).sum(axis=1) / mask.sum(axis=1)
            output = output.astype(np.float32)  # type: ignore
            if normalize:
                norms = np.linalg.norm(output, axis=1, keepdims=True)
                output = output / np.clip(norms, 1e-9, None)

            all_embeddings.append(output.astype(np.float32))  # type: ignore

        embeddings = np.concatenate(all_embeddings, axis=0)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-9, None)
        return embeddings


class PoolingStrategy:
    MEAN = "mean"
    MAX = "max"
    WEIGHTED = "weighted"


class ChunkedEncoder:
    """
    对超长文本分块编码。优先使用模型 tokenizer 进行 token 级分块，
    避免按字符粗暴切分导致语义断裂。
    """
    def __init__(self, st_model, max_tokens=120, overlap=20, strategy=PoolingStrategy.MEAN):
        self.st = st_model
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.strategy = strategy
        # 尝试获取 tokenizer 用于精确分块
        self.tokenizer = getattr(st_model, 'tokenizer', None)

    def _chunk(self, text: str):
        if self.tokenizer is not None:
            # 使用 tokenizer 进行 token 级分块
            tokens = self.tokenizer.tokenize(text)
            step = self.max_tokens - self.overlap
            chunks = [tokens[i:i+self.max_tokens] for i in range(0, len(tokens), step)]
            # 将 token 列表转回文本
            return [text]
        else:
            # 按字符（中文）或空格（纯英文）分块
            tokens = text.split() if " " in text and not re.search(r"[\u4e00-\u9fff]", text) else list(text)
            step = self.max_tokens - self.overlap
            return [tokens[i:i+self.max_tokens] for i in range(0, len(tokens), step)]

    def encode(self, text: str) -> np.ndarray:
        chunks = self._chunk(text)
        if not chunks:
            return np.zeros(self.st.get_sentence_embedding_dimension(), dtype=np.float32)

        if self.tokenizer is not None:
            chunk_texts = ["".join(c) for c in chunks]
        else:
            chunk_texts = ["".join(c) for c in chunks]

        vecs = self.st.encode(
            chunk_texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False
        ).astype(np.float32)

        if self.strategy == PoolingStrategy.MEAN:
            return vecs.mean(axis=0)
        elif self.strategy == PoolingStrategy.MAX:
            return vecs.max(axis=0)
        elif self.strategy == PoolingStrategy.WEIGHTED:
            weights = np.array([
                1.0 + 0.3 * ("!" in ct or "！" in ct) + 0.2 * ("?" in ct or "？" in ct)
                for ct in chunk_texts
            ], dtype=np.float32)
            weights = weights / weights.sum()
            return (weights[:, None] * vecs).sum(axis=0)
        else:
            raise ValueError(f"未知池化策略: {self.strategy}")


class EmotionDataset(Dataset):
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
