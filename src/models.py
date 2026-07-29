import re
from torch import nn,exp
from torch.utils.data import Dataset
import numpy as np

class EmotionClassifierNet(nn.Module):
    """轻量多语言情感分类头。"""
    def __init__(self, input_dim=384, num_classes=19, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, num_classes)),
            nn.BatchNorm1d(max(hidden_dim // 2, num_classes)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, num_classes), num_classes),
        )

    def forward(self, x):
        return self.net(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha          # 类别权重 tensor，形状 [num_classes]
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.alpha, reduction='none'
        )
        pt = exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

#池化策略类型
class PoolingStrategy:
    MEAN = "mean"
    MAX = "max"
    WEIGHTED = "weighted"

class ChunkedEncoder:
    """
    对超长文本按 max_tokens 分块，分别编码后池化为单一 384 维向量。
    支持 mean / max / weighted 三种池化策略。
    """
    def __init__(
        self,
        st_model,
        max_tokens=120,
        overlap=20,
        strategy=PoolingStrategy.MEAN
    ):
        self.st = st_model
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.strategy = strategy

    def _chunk(self, text: str):
        """按近似 token 长度分块（中文 1.5 字/token，英文 0.75 词/token）。"""
        # 用空格粗略切英文词，中文按字符
        tokens = text.split() if " " in text and not re.search(r"[\u4e00-\u9fff]", text) else list(text)
        step = self.max_tokens - self.overlap
        return [tokens[i:i+self.max_tokens] for i in range(0, len(tokens), step)]

    def encode(self, text: str) -> np.ndarray:
        import re
        chunks = self._chunk(text)
        if not chunks:
            return np.zeros(self.st.get_sentence_embedding_dimension(), dtype=np.float32)
        vecs = self.st.encode(
            ["".join(c) for c in chunks],
            convert_to_numpy=True, 
            normalize_embeddings=False,
            show_progress_bar=False
        ).astype(np.float32)  # (n_chunks, 384)

        if self.strategy == PoolingStrategy.MEAN:
            return vecs.mean(axis=0)
        elif self.strategy == PoolingStrategy.MAX:
            return vecs.max(axis=0)
        elif self.strategy == PoolingStrategy.WEIGHTED:
            # 启发式：含情感标点/强调词的块权重更高
            weights = np.array([
                1.0 + 0.3 * ("!" in "".join(c) or "！" in "".join(c))
                   + 0.2 * ("?" in "".join(c) or "？" in "".join(c))
                for c in chunks
            ], dtype=np.float32)
            weights = weights / weights.sum()
            return (weights[:, None] * vecs).sum(axis=0)
        else:
            raise ValueError(f"未知池化策略: {self.strategy}")


class EmotionDataset(Dataset):
    """
    存储文本 + 标签 +预计算 embedding。
    继承 torch Dataset，可直接用于 DataLoader。
    """
    def __init__(self, texts, labels, embeddings=None):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.embeddings = embeddings  # numpy array or None

    def set_embeddings(self, emb: np.ndarray):
        self.embeddings = emb.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.embeddings is not None:
            return self.embeddings[idx], self.labels[idx]
        return self.texts[idx], self.labels[idx]
