"""模型定义（v2.0）。

新增：
  * build_encoder_lora   —— XLM-R + LoRA（路径 A）
  * XlmrTorchEncoder     —— 训练期 PyTorch 编码器（XLM-R）
  * PromptMaskEncoder    —— 冻结编码器 + prompt 完形（路径 B）
  * merge_lora           —— 训练后合并 LoRA，产出标准 XLM-R 结构
保留：
  * LightEmotionClassifier / FocalLoss / ONNXEncoder / ChunkedEncoder / EmotionDataset
"""

import os
import warnings
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from .config import Config, PoolingStrategy


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
    """轻量情感分类头：投影层 + 单层残差块 + 线性分类。

    v2：input_dim 默认由 384 改为 768（XLM-R hidden_size）。
    """

    def __init__(
        self,
        input_dim=768,
        num_classes=18,
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
    """Focal Loss + 类别权重 + 标签平滑。"""

    def __init__(self, gamma=2.0, weight=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, sample_weight: Optional[torch.Tensor] = None):
        log_probs = F.log_softmax(logits, dim=-1)
        pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1).exp()
        ce = F.nll_loss(log_probs, targets, weight=self.weight,
                        reduction='none', ignore_index=-100)
        if self.weight is not None:
            w = self.weight.gather(0, targets).clamp_min(1e-6)
            ce = ce / w * w.detach()
        loss = ((1 - pt) ** self.gamma) * ce

        if self.label_smoothing > 0:
            smooth = -log_probs.mean(dim=-1) * self.label_smoothing
            loss = loss * (1 - self.label_smoothing) + smooth

        if sample_weight is not None:          # v2：MT 样本降权
            loss = loss * sample_weight.to(loss.dtype)

        if self.reduction == 'mean':
            return loss.sum() / max(1.0, float(
                sample_weight.sum().item() if sample_weight is not None else loss.numel()
            ))
        if self.reduction == 'sum':
            return loss.sum()
        return loss


# ══════════════════════════════════════════════════════════════════
# XLM-R 编码器
# ══════════════════════════════════════════════════════════════════

def load_tokenizer(model_name: Optional[str] = None, tokenizer_dir: Optional[str] = None):
    """加载 XLM-R tokenizer。

    优先级：显式 tokenizer_dir > 训练基座目录（三语裁剪产物自带）> model_name。

    注意：XLM-R 使用 SentencePiece 慢分词器；词表裁剪后必须同步重写
    sentencepiece.bpe.model，否则仍会产出已删除的 id（静默出错）。
    """
    from transformers import AutoTokenizer
    name = model_name or Config.MODEL_NAME
    for cand in (tokenizer_dir, Config.tokenizer_dir_for_train()):
        if cand and os.path.isdir(cand) and any(
                f.startswith("tokenizer") or f.startswith("sentencepiece")
                or f == "spiece.model" for f in os.listdir(cand)):
            try:
                return AutoTokenizer.from_pretrained(cand, local_files_only=True)
            except Exception:
                break
    return AutoTokenizer.from_pretrained(name)


def _xlmr_position_ids(input_ids: torch.Tensor, padding_idx: int = 1) -> torch.Tensor:
    """复现 HF create_position_ids_from_input_ids。

    XLM-R 的 position embedding 从 padding_idx + 1 = 2 起算。
    自实现前向（尤其 Rust 侧）必须复现该逻辑，否则右 padding 的短句位置编码全错。
    """
    mask = input_ids.ne(padding_idx).int()
    incremental = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental.long() + padding_idx


def build_encoder_lora(
    num_labels: int,
    model_name: Optional[str] = None,
    r: Optional[int] = None,
    alpha: Optional[float] = None,
    dropout: Optional[float] = None,
    unfreeze_from: Optional[int] = None,
    target_modules: Optional[Sequence[str]] = None,
):
    """XLM-R + LoRA（路径 A）。

    * 全部 encoder 基础权重先冻结；
    * 只在 layer[unfreeze_from:] 上插入 LoRA 适配器（query/value）；
    * 分类头用 HF 自带 classifier（num_labels），可直接接 CE/Focal。

    注意：HF 的 XLMRobertaSelfAttention 用 self.query / self.key / self.value，
    不是 q_proj / v_proj —— 写错会静默退化成「零个可训练模块」。
    """
    try:
        from transformers import XLMRobertaForSequenceClassification
        from peft import LoraConfig, get_peft_model
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "路径 A 需要 transformers + peft：pip install 'transformers>=4.30' 'peft>=0.10'"
        ) from e

    name = model_name or Config.MODEL_NAME
    r = Config.LORA_R if r is None else r
    alpha = Config.LORA_ALPHA if alpha is None else alpha
    dropout = Config.LORA_DROPOUT if dropout is None else dropout
    unfreeze_from = Config.LORA_UNFREEZE_FROM if unfreeze_from is None else unfreeze_from
    target_modules = list(target_modules or Config.LORA_TARGET_MODULES)

    model = XLMRobertaForSequenceClassification.from_pretrained(
        name,
        num_labels=num_labels,
        hidden_dropout_prob=Config.DROPOUT,
        attention_probs_dropout_prob=Config.DROPOUT,
    )

    num_layers = len(model.roberta.encoder.layer)
    if unfreeze_from >= num_layers:
        raise ValueError(f"LORA_UNFREEZE_FROM={unfreeze_from} 超出层数 {num_layers}")

    for p in model.roberta.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=list(target_modules),
        layers_to_transform=list(range(unfreeze_from, num_layers)),
    )
    model = get_peft_model(model, lora_cfg)

    # 分类头与 pooler 始终可训
    for p in model.base_model.model.classifier.parameters():
        p.requires_grad = True
    return model


def merge_lora(model) -> "torch.nn.Module":
    """合并 LoRA 权重并卸载 peft 包装，得到标准 XLM-R 结构。

    必须在剪枝 / ONNX 导出之前调用。
    """
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    return model


def build_param_groups(model, base_model=None, lora_lr=None, head_lr=None, llrd=None):
    """分组学习率：LoRA / head 用较大 lr，必要时对层做 LLRD。

    返回 [(params, lr), ...]，可直接喂给 AdamW。
    """
    lora_lr = Config.LORA_LR if lora_lr is None else lora_lr
    head_lr = Config.HEAD_LR if head_lr is None else head_lr
    llrd = Config.LLRD if llrd is None else llrd

    lora_params, head_params, other = [], [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(p)
        elif "classifier" in name or "score" in name:
            head_params.append(p)
        else:
            other.append(p)
        seen.add(name)

    groups = []
    if Config.USE_LLRD and lora_params:
        # 按 layer 下标衰减：越靠后（越接近输出）lr 越大
        import re
        layer_lr = {}
        for name, p in model.named_parameters():
            if "lora_" not in name or not p.requires_grad:
                continue
            m = re.search(r"layer\.(\d+)\.", name)
            idx = int(m.group(1)) if m else 0
            layer_lr.setdefault(idx, []).append(p)
        n_layers = max(layer_lr) + 1 if layer_lr else 1
        for idx, ps in sorted(layer_lr.items()):
            groups.append({"params": ps, "lr": lora_lr * (llrd ** (n_layers - 1 - idx))})
    elif lora_params:
        groups.append({"params": lora_params, "lr": lora_lr})

    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    if other:
        groups.append({"params": other, "lr": lora_lr})
    return groups


class XlmrTorchEncoder:
    """训练期 PyTorch 编码器：输出 [CLS] / mean / cls+mean 池化后的向量。

    仅用于 cached 模式（冻结编码器预编码）。e2e 模式直接端到端前向。
    """

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu",
                 max_len: Optional[int] = None, strategy: Optional[str] = None,
                 tokenizer_dir: Optional[str] = None, use_lang_token: Optional[bool] = None):
        from transformers import XLMRobertaModel
        name = model_name or Config.MODEL_NAME
        tokenizer_dir = tokenizer_dir or Config.tokenizer_dir_for_train()
        self.model = XLMRobertaModel.from_pretrained(name).to(device)
        self.device = device
        self.max_len = max_len or Config.MAX_SEQ_LEN
        self.strategy = strategy or Config.POOL_STRATEGY
        self.use_lang_token = Config.USE_LANG_TOKEN if use_lang_token is None else use_lang_token
        self.tokenizer = load_tokenizer(name, tokenizer_dir)
        if self.use_lang_token:
            self._ensure_lang_tokens()
        self.model.eval()

    def _ensure_lang_tokens(self):
        """把 <zh>/<en>/<ja> 加进词表并同步 resize embedding。

        新增 token 的 embedding 用 <mask> 的向量初始化，随训练更新。
        """
        new_tokens = [t for t in Config.LANG_TOKENS.values()
                      if t not in self.tokenizer.get_vocab()]
        if not new_tokens:
            return
        self.tokenizer.add_tokens(new_tokens, special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer),mean_resizing=False)
        with torch.no_grad():
            emb = self.model.get_input_embeddings().weight
            anchor_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id
            anchor = emb[anchor_id].detach().clone()
            for t in new_tokens:
                emb[self.tokenizer.convert_tokens_to_ids(t)] = anchor

    def _prepare(self, texts: Sequence[str], lang: Optional[Sequence[str]] = None) -> list:
        out = []
        for i, t in enumerate(texts):
            if self.use_lang_token and lang is not None:
                tok = Config.LANG_TOKENS.get(str(lang[i]).lower(), "")
                out.append(f"{tok}{t}" if tok else t)
            else:
                out.append(t)
        return out

    @torch.no_grad()
    def encode(self, texts: Sequence[str], lang: Optional[Sequence[str]] = None,
               batch_size: int = 32, normalize: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if isinstance(t, str) else "" for t in texts]
        if not texts:
            return np.zeros((0, Config.INPUT_DIM), dtype=np.float32)

        prepared = self._prepare(texts, lang)
        dim = Config.INPUT_DIM * 2 if self.strategy == PoolingStrategy.CLS_MEAN else Config.INPUT_DIM
        out = np.zeros((len(texts), dim), dtype=np.float32)

        order = sorted(range(len(prepared)), key=lambda i: len(prepared[i]))
        for s in range(0, len(order), batch_size):
            idxs = order[s:s + batch_size]
            batch = [prepared[i] for i in idxs]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_len, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            hidden = self.model(**enc).last_hidden_state          # [B, L, H]
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            cls = hidden[:, 0, :]
            vec = {"cls": cls,
                   "mean": mean,
                   "cls+mean": torch.cat([cls, mean], dim=-1)}.get(self.strategy, cls)
            vec = vec.float().cpu().numpy()
            if normalize:
                n = np.linalg.norm(vec, axis=1, keepdims=True)
                vec = vec / np.clip(n, 1e-9, None)
            for j, i in enumerate(idxs):
                out[i] = vec[j]
        return out

    def to(self, device: str):
        self.device = device
        self.model = self.model.to(device)
        return self


class PromptMaskEncoder:
    """路径 B：冻结编码器 + prompt 完形填空，取 [MASK] 位置的 hidden state。

    模板固定 ⇒ [MASK] 位置的向量可离线预计算并缓存，
    因此可完整复用 v1 的 stage_embed -> embeddings.npz -> stage_train 两段式。
    """

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu",
                 tokenizer_dir: Optional[str] = None, max_len: Optional[int] = None):
        from transformers import XLMRobertaModel
        name = model_name or Config.MODEL_NAME
        tokenizer_dir = tokenizer_dir or Config.tokenizer_dir_for_train()
        self.model = XLMRobertaModel.from_pretrained(name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.max_len = max_len or Config.MAX_SEQ_LEN
        self.tokenizer = load_tokenizer(name, tokenizer_dir)
        self.templates = Config.PROMPT_TEMPLATES

    @staticmethod
    def detect_lang(text: str) -> str:
        han = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        kana = sum(1 for c in text if "\u3040" <= c <= "\u30ff")
        if kana > 0:
            return "ja"
        if han > 0:
            return "zh"
        return "en"

    def _templated(self, text: str, lang: Optional[str]) -> str:
        lang = lang or self.detect_lang(text)
        tpl = self.templates.get(lang, self.templates["en"])
        return tpl.format(text=text, mask=self.tokenizer.mask_token or "<mask>")

    @torch.no_grad()
    def encode(self, texts: Sequence[str], lang: Optional[Sequence[str]] = None,
               batch_size: int = 32, normalize: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, Config.INPUT_DIM), dtype=np.float32)

        langs = list(lang) if lang is not None else [None] * len(texts)
        templated = [self._templated(t, langs[i]) for i, t in enumerate(texts)]
        mask_token = self.tokenizer.mask_token or "<mask>"
        mask_id = self.tokenizer.mask_token_id

        out = np.zeros((len(texts), Config.INPUT_DIM), dtype=np.float32)
        order = sorted(range(len(templated)), key=lambda i: len(templated[i]))
        for s in range(0, len(order), batch_size):
            idxs = order[s:s + batch_size]
            enc = self.tokenizer([templated[i] for i in idxs], padding=True, truncation=True,
                                 max_length=self.max_len, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            hidden = self.model(**enc).last_hidden_state
            for j, i in enumerate(idxs):
                pos = int((enc["input_ids"][j] == mask_id).nonzero(as_tuple=True)[0][0])
                out[i] = hidden[j, pos, :].float().cpu().numpy()
        if normalize:
            n = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.clip(n, 1e-9, None)
        return out


# ══════════════════════════════════════════════════════════════════
# ONNX 编码器（语义改为 XLM-R）
# ══════════════════════════════════════════════════════════════════

class ONNXEncoder:
    """XLM-R ONNX 编码器 + 池化。

    v2 变化：
      * 默认 tokenizer 由 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        改为 xlm-roberta-base；
      * 不再需要 token_type_ids（XLM-R 不做 NSP）；
      * 分块逻辑默认关闭（数据 max=45 字符，无长文本场景）。
    """

    def __init__(
        self,
        onnx_path: str,
        tokenizer_dir: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        overlap: int = 20,
        pooling: Optional[str] = None,
        enable_chunking: bool = False,
    ):
        import onnxruntime as ort

        self.max_tokens = max_tokens or Config.MAX_SEQ_LEN
        self.overlap = overlap
        self.enable_chunking = enable_chunking
        self.pooling = pooling or Config.POOL_STRATEGY
        tokenizer_name = tokenizer_name or Config.MODEL_NAME

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = max(1, min(8, os.cpu_count() or 4))
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs_info = {i.name: i for i in self.sess.get_inputs()}
        self.output_name = self.sess.get_outputs()[0].name
        # XLM-R：pad_id 必须是 1（不是 0）
        self.tokenizer = load_tokenizer(tokenizer_name, tokenizer_dir)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1

    def _chunk_ids(self, text: str) -> List[List[int]]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        limit = max(1, self.max_tokens - 2)
        if len(ids) <= limit or not self.enable_chunking:
            return [ids[:limit]]
        step = max(1, limit - self.overlap)
        chunks = [ids[i:i + limit] for i in range(0, len(ids), step)]
        if len(chunks) > 1 and len(chunks[-1]) < max(1, limit // 4):
            chunks[-2].extend(chunks[-1])
            chunks.pop()
        return chunks

    def _pack(self, id_chunks: Sequence[Sequence[int]]) -> dict:
        cls_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else 0
        sep_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else 2
        seqs = [[cls_id] + list(c) + [sep_id] for c in id_chunks]
        max_len = max(len(s) for s in seqs)
        input_ids = np.full((len(seqs), max_len), self.pad_id, dtype=np.int64)
        attention = np.zeros((len(seqs), max_len), dtype=np.int64)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = s
            attention[i, :len(s)] = 1
        enc = {"input_ids": input_ids, "attention_mask": attention}
        # XLM-R 不需要 token_type_ids，但若导出图带了该输入仍要喂
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
        if out.ndim == 3:  # type: ignore
            mask = enc["attention_mask"][:, :, None].astype(np.float32)
            mean = (out * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            cls = out[:, 0, :]
            if self.pooling == PoolingStrategy.CLS:
                out = cls
            elif self.pooling == PoolingStrategy.CLS_MEAN:
                out = np.concatenate([cls, mean], axis=-1)
            else:
                out = mean
        return out.astype(np.float32)  # type: ignore

    def encode(self, texts, lang=None, batch_size=32, normalize=False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = [t if isinstance(t, str) else "" for t in texts]
        if not texts:
            return np.zeros((0, Config.INPUT_DIM), dtype=np.float32)

        if lang is not None and Config.USE_LANG_TOKEN:
            texts = [f"{Config.LANG_TOKENS.get(str(lang[i]).lower(), '')}{t}"
                     for i, t in enumerate(texts)]

        flat: List[List[int]] = []
        spans: List[tuple] = []
        for i, t in enumerate(texts):
            chunks = self._chunk_ids(t)
            spans.append((i, len(chunks)))
            for c in chunks:
                flat.append(c)

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


class ChunkedEncoder:
    """[DEPRECATED] 超长文本分块编码器（MiniLM/SentenceTransformer 专用）。

    v2 已换 XLM-R，且数据 max=45 字符无长文本场景；保留仅为兼容旧脚本。
    """

    def __init__(self, st_model, max_tokens=120, overlap=20, strategy=PoolingStrategy.MEAN):
        warnings.warn("ChunkedEncoder 已废弃（XLM-R 场景请使用 XlmrTorchEncoder）",
                      DeprecationWarning, stacklevel=2)
        self.st = st_model
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.strategy = strategy
        self.tokenizer = getattr(st_model, "tokenizer", None)

    def encode(self, text: str) -> np.ndarray:  # pragma: no cover - 兼容路径
        chunks = [text]
        vecs = self.st.encode(chunks, convert_to_numpy=True, normalize_embeddings=False,
                              show_progress_bar=False).astype(np.float32)
        if self.strategy == PoolingStrategy.MAX:
            return vecs.max(axis=0)
        if self.strategy == PoolingStrategy.WEIGHTED:
            w = np.ones(len(vecs), dtype=np.float32)
            w /= w.sum()
            return (w[:, None] * vecs).sum(axis=0)
        return vecs.mean(axis=0)


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


class EmotionDataset(Dataset):
    """惰性数据集：已有 embedding 时直接取向量，否则返回原文（e2e 用）。"""

    def __init__(self, texts, labels, embeddings=None, sample_weight=None):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.embeddings = embeddings
        self.sample_weight = (None if sample_weight is None
                              else np.asarray(sample_weight, dtype=np.float32))

    def set_embeddings(self, emb: np.ndarray):
        self.embeddings = emb.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.embeddings is not None:
            out = [self.embeddings[idx], self.labels[idx]]
        else:
            out = [self.texts[idx], self.labels[idx]]
        if self.sample_weight is not None:
            out.append(self.sample_weight[idx])
        return tuple(out)


class TextClassificationDataset(Dataset):
    """e2e 模式（LoRA / 全量微调）的文本级数据集。

    v1 的 embeddings.npz 缓存在编码器参与训练后即失效，故 e2e 必须走文本级。
    """

    def __init__(self, texts, labels, tokenizer, max_len=None, lang=None,
                 use_lang_token=None, sample_weight=None):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tok = tokenizer
        self.max_len = max_len or Config.MAX_SEQ_LEN
        self.lang = list(lang) if lang is not None else None
        self.use_lang_token = Config.USE_LANG_TOKEN if use_lang_token is None else use_lang_token
        self.sample_weight = (None if sample_weight is None
                              else np.asarray(sample_weight, dtype=np.float32))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        t = self.texts[i]
        if self.use_lang_token and self.lang is not None:
            tok = Config.LANG_TOKENS.get(str(self.lang[i]).lower(), "")
            t = f"{tok}{t}" if tok else t
        enc = self.tok(t, truncation=True, max_length=self.max_len, add_special_tokens=True)
        item = {"input_ids": enc["input_ids"],
                "attention_mask": enc.get("attention_mask", [1] * len(enc["input_ids"])),
                "labels": int(self.labels[i])}
        if self.sample_weight is not None:
            item["sample_weight"] = float(self.sample_weight[i])
        return item


def make_collate(pad_id: int = 1, pad_to_multiple_of: Optional[int] = None):
    """动态 padding collate（按 batch 内最长，可选对齐到 multiple_of）。"""
    def collate(batch: list) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        if pad_to_multiple_of and pad_to_multiple_of > 1:
            max_len = int(np.ceil(max_len / pad_to_multiple_of) * pad_to_multiple_of)
        ids = np.full((len(batch), max_len), pad_id, dtype=np.int64)
        mask = np.zeros((len(batch), max_len), dtype=np.int64)
        for i, b in enumerate(batch):
            n = min(len(b["input_ids"]), max_len)
            ids[i, :n] = b["input_ids"][:n]
            mask[i, :n] = 1
        out = {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(mask),
            "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
        }
        # 长度分桶：批内按 token 数降序，便于外层做 bucketing
        out["lengths"] = torch.tensor([min(len(b["input_ids"]), max_len) for b in batch])
        if "sample_weight" in batch[0]:
            out["sample_weight"] = torch.tensor(
                [float(b.get("sample_weight", 1.0)) for b in batch], dtype=torch.float)
        return out
    return collate
