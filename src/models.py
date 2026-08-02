import os
import re
from typing import Optional
from torch.utils.data import Dataset
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class ArcMarginProduct(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        s=30.0,
        m=0.50
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x, label=None):
        cosine = F.linear(
            F.normalize(x),
            F.normalize(self.weight)
        )
        if label is None:
            return cosine * self.s
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(
            1,
            label.view(-1,1),
            1
        )
        theta = torch.acos(
            torch.clamp(cosine,-1+1e-7,1-1e-7)
        )
        target = torch.cos(theta+self.m)
        logits = cosine*(1-one_hot)+target*one_hot
        return logits*self.s

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))

class ProjectionHead(nn.Module):
    """投影头。"""
    def __init__(
            self,
            input_dim=384,
            proj_dim=256,
            dropout=0.2
        ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, proj_dim)
        )

    def forward(self,x):
        x=self.net(x)
        return nn.functional.normalize(x,p=2,dim=-1)

class PrototypeClassifier(nn.Module):
    """轻量多语言原型分类头。"""
    def __init__(
        self,
        feat_dim,
        num_classes,
        temperature=16.0
    ):
        super().__init__()
        self.prototype = nn.Parameter(
            torch.randn(
                num_classes,
                feat_dim
            )
        )
        nn.init.xavier_uniform_(self.prototype)
        self.temperature=temperature

    def forward(self,x):
        x=F.normalize(x,p=2,dim=1)
        p=F.normalize(
            self.prototype,
            p=2,
            dim=1
        )
        logits=torch.matmul(
            x,
            p.t()
        )
        return logits*self.temperature

class EmotionClassifier(nn.Module):
    """轻量多语言情感分类头。"""
    def __init__(
        self,
        input_dim=384,
        num_classes=19,
        hidden_dim=384,
        proj_dim=256,
        dropout=0.3
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.Sequential(
            ResidualBlock(hidden_dim,dropout),
            ResidualBlock(hidden_dim,dropout)
        )
        self.projector = ProjectionHead(
            hidden_dim,
            proj_dim,
            dropout
        )
        self.classifier = PrototypeClassifier(
            proj_dim,
            num_classes
        )
        self.arcface = ArcMarginProduct(
            proj_dim,
            num_classes
        )

    def forward(self,x,label=None):
        x = self.input_proj(x)
        x = self.encoder(x)
        feat = self.projector(x)
        logits = self.classifier(feat)
        if label is not None:
            arc_logits = self.arcface(feat, label)  # 带标签时返回ArcFace logits
        else:
            arc_logits = self.arcface(feat)         # 无标签时返回常规cosine
        return logits, feat, arc_logits


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, margin=0.3, topk_ratio=0.7, smoothing=0.1, weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.topk_ratio = topk_ratio
        self.smoothing = smoothing
        self.weights = weights
        self.ce = nn.CrossEntropyLoss(label_smoothing=smoothing, reduction='none', weight=weights)
    
    def forward(self, logits, labels, features):
        # 1. Label Smoothing CE Loss (per sample)
        ce_loss = self.ce(logits, labels)  # shape: (batch,)

        # 2. Margin Ranking Loss (拉大正确类与最高错误类之间的差距)
        batch_size = logits.size(0)
        correct_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)  # (batch,)
        # 获取最高错误类的 logits
        wrong_logits = logits.clone()
        wrong_logits.scatter_(1, labels.unsqueeze(1), -1e9)
        max_wrong, _ = wrong_logits.max(dim=1)  # (batch,)
        margin_loss = F.relu(self.margin - (correct_logits - max_wrong))  # hinge-like

        # 3. 合并损失，取 top70%
        combined = ce_loss + 0.5 * margin_loss  # 加权系数可调
        k = max(1, int(batch_size * self.topk_ratio))
        topk_values, _ = torch.topk(combined, k, largest=True)
        final_loss = topk_values.mean()

        return final_loss

class ONNXEncoder:
    """
    用 ONNX Runtime 加载量化后的 SentenceTransformer 编码器。
    输入: 文本列表
    输出: 384 维句向量 (numpy float32)
    """
    def __init__(
            self,
            onnx_path: str,
            tokenizer_dir: Optional[str] = None,
            tokenizer_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        # 获取模型所有输入的元信息
        self.inputs_info = {i.name: i for i in self.sess.get_inputs()}
        self.output_name = self.sess.get_outputs()[0].name

        # 加载 tokenizer
        if tokenizer_dir and os.path.isdir(tokenizer_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir, 
                local_files_only=True
            )
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
                padding=True, truncation=True, max_length=128,
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
                        raise ValueError(f"ONNX 模型要求未知输入 '{input_name}'，请检查模型导出配置。")

            output = self.sess.run([self.output_name], feeds)[0]

            # 如果输出是三维（batch, seq_len, dim），做 mean pooling
            if output.ndim == 3: # type: ignore
                mask = enc["attention_mask"][:, :, None].astype(np.float32)
                output = (output * mask).sum(axis=1) / mask.sum(axis=1)
            output = output.astype(np.float32) # type: ignore
            if normalize:
                norms = np.linalg.norm(output, axis=1, keepdims=True)
                output = output / np.clip(norms, 1e-9, None)

            all_embeddings.append(output.astype(np.float32)) # type: ignore

        embeddings = np.concatenate(all_embeddings, axis=0)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-9, None)
        return embeddings

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
