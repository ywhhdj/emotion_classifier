"""
端到端 LoRA 微调（stage: finetune）。

与 head-only 训练（stage_train，基于缓存嵌入）的区别：
  · stage_train  : 嵌入预先算好，只训分类头。快，但编码器不动 → LoRA 无从训练。
  · stage_finetune: 每个 batch 都过一遍 XLM-R，编码器（LoRA）+ 分类头联合更新。
                   这是 LoRA 真正被训练的地方，代价是每轮都要重算前向。

因此默认轮数远少于 head-only（FINETUNE_EPOCHS=12 vs EPOCHS=120），
并建议先用 stage_train 把分类头训好再进来做微调（warm start）。

冻结策略（按需求）：
  · layer 0..7  : 完全冻结，不插 LoRA、不求梯度
  · layer 8..11 : query/value 插 LoRA(rank=8)，基座权重冻结，仅适配器可训练
  · 其余（embedding / LayerNorm / FFN）: 冻结
"""
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config  # noqa: E402
from .lora import lora_state_dict, set_lora_merged  # noqa: E402
from .models import EmotionNet, FocalLoss, XLMRobertaLoRAEncoder  # noqa: E402
from .utils import class_weights, macro_f1, mixup, set_seed  # noqa: E402


# ───────────────────────── 数据 ─────────────────────────
class TextDataset(Dataset):
    """返回原文，tokenize 放到 collate 里做（动态 padding，省算力）。"""

    def __init__(self, texts: Sequence[str], labels: Sequence[int]):
        self.texts = [str(t) for t in texts]
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], int(self.labels[idx])


def _make_collate(tokenizer, max_length: int):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long)
    return collate


# ───────────────────────── 构建 ─────────────────────────
def build_model(args, num_labels: int, verbose: bool = True) -> EmotionNet:
    """构建 XLM-R(+LoRA) + 分类头。

    若已存在 head 权重（stage_train 产出）则热启动分类头，避免从零学。
    """
    encoder = XLMRobertaLoRAEncoder(
        model_name=Config.MODEL_NAME,
        lora_rank=getattr(args, "lora_rank", Config.LORA_RANK),
        lora_alpha=getattr(args, "lora_alpha", Config.LORA_ALPHA),
        lora_dropout=getattr(args, "lora_dropout", Config.LORA_DROPOUT),
        lora_targets=getattr(args, "lora_targets", Config.LORA_TARGETS) or Config.LORA_TARGETS,
        freeze_layers=getattr(args, "freeze_layers", Config.FREEZE_LAYERS),
        lora_layers=getattr(args, "lora_layers", Config.LORA_LAYERS),
        max_length=getattr(args, "max_seq_len", Config.MAX_SEQ_LEN),
        train_layernorm=getattr(args, "train_layernorm", Config.TRAIN_LAYERNORM),
        verbose=verbose,
    )
    model = EmotionNet(
        encoder,
        num_labels=num_labels,
        hidden_dim=getattr(args, "hidden_dim", Config.HIDDEN_DIM),
        dropout=getattr(args, "dropout", Config.DROPOUT),
    )

    # 热启动分类头（head-only 阶段训好的权重）
    ckpt = Config.MODEL_PATH
    if ckpt and os.path.exists(ckpt):
        try:
            state = torch.load(ckpt, map_location="cpu")
            head_state = state.get("state_dict", state)
            # 兼容旧格式：直接就是 state_dict
            if head_state and all(not k.startswith("head.") for k in head_state):
                head_state = {f"head.{k}": v for k, v in head_state.items()}
            missing = model.head.load_state_dict(
                {k.replace("head.", "", 1): v for k, v in head_state.items()
                 if k.startswith("head.")}, strict=False)
            if verbose:
                print(f"[finetune] 已热启动分类头 ← {ckpt}")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[finetune] 分类头热启动跳过（{e}），从零初始化")

    # 可选：加载已有 LoRA 适配器继续训练
    if getattr(args, "lora_ckpt", None) and os.path.exists(args.lora_ckpt):
        state = torch.load(args.lora_ckpt, map_location="cpu")
        model.encoder.bert.load_state_dict(state, strict=False)
        if verbose:
            print(f"[finetune] 已加载 LoRA 适配器 ← {args.lora_ckpt}")

    return model


def _param_groups(model: EmotionNet, lr: float, head_lr: float, wd: float):
    """LoRA 与分类头用不同学习率；其余参数本就冻住，不会进优化器。"""
    lora_params, head_params, other = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_A" in name or "lora_B" in name:
            lora_params.append(p)
        elif name.startswith("head."):
            head_params.append(p)
        else:
            other.append(p)
    groups = [
        {"params": lora_params, "lr": lr, "weight_decay": wd},
        {"params": head_params, "lr": head_lr, "weight_decay": wd},
    ]
    if other:
        groups.append({"params": other, "lr": lr, "weight_decay": wd})
    return groups


# ───────────────────────── 评估 ─────────────────────────
@torch.no_grad()
def evaluate(
    model: EmotionNet,
    loader: DataLoader,
    device: torch.device,
    num_labels: int
) -> Dict:
    model.eval()
    ys, ps = [], []
    for input_ids, mask, y in loader:
        input_ids, mask = input_ids.to(device), mask.to(device)
        logits = model(input_ids, mask)
        ps.append(logits.argmax(-1).cpu().numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = float((y_true == y_pred).mean())
    return {"accuracy": acc, "macro_f1": macro_f1(y_true, y_pred, num_labels),
            "y_true": y_true, "y_pred": y_pred}


# ───────────────────────── 主流程 ─────────────────────────
def stage_finetune(
    args,
    split: Optional[dict] = None,
    num_labels: Optional[int] = None,
    id2label: Optional[dict] = None
) -> Dict:
    set_seed(getattr(args, "seed", Config.SEED))
    if split is None:
        split = {
            k: pd.read_csv(os.path.join(Config.OUTPUT_PATH, f"{k}.csv")) #type: ignore
            for k in ["train", "val"]
        }

    if num_labels is None:
        num_labels = int(split["train"]["label_id"].max()) + 1

    device = torch.device("cuda" if (getattr(args, "device", "cpu") == "cuda"
                                     and torch.cuda.is_available()) else "cpu")
    print(f"[finetune] device: {device}")

    model = build_model(args, num_labels).to(device)

    # ── 数据 ──
    max_len = getattr(args, "max_seq_len", Config.MAX_SEQ_LEN)
    collate = _make_collate(model.encoder.tokenizer, max_len)
    bs = getattr(args, "finetune_batch", Config.FINETUNE_BATCH)
    train_loader = DataLoader(
        TextDataset(split["train"]["text"], split["train"]["label_id"]),
        batch_size=bs, shuffle=True, collate_fn=collate, num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        TextDataset(split["val"]["text"], split["val"]["label_id"]),
        batch_size=bs * 2, shuffle=False, collate_fn=collate, num_workers=0,
    ) if "val" in split and len(split["val"]) else None

    # ── 损失：Focal + 类别权重 ──
    counts = np.bincount(np.asarray(split["train"]["label_id"]), minlength=num_labels)
    weights = class_weights(counts, Config.CLASS_WEIGHT, Config.EFFECTIVE_BETA)
    w_t = torch.tensor(weights, dtype=torch.float32, device=device)
    gamma = getattr(args, "focal_gamma", Config.FOCAL_GAMMA)
    use_focal = (not getattr(args, "no_focal", False)) and Config.USE_FOCAL_LOSS
    criterion = FocalLoss(gamma=gamma if use_focal else 0.0, weight=w_t)

    # ── 优化器：LoRA / head 分组学习率 ──
    lr = getattr(args, "finetune_lr", Config.FINETUNE_LR)
    head_lr = getattr(args, "finetune_head_lr", Config.FINETUNE_HEAD_LR)
    wd = getattr(args, "weight_decay", Config.WEIGHT_DECAY)
    groups = _param_groups(model, lr, head_lr, wd)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[finetune] 可训练参数 {n_trainable:,} "
          f"（LoRA lr={lr}, head lr={head_lr}）")
    if n_trainable == 0:
        raise RuntimeError("没有可训练参数：LoRA 注入或冻结策略有误")

    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    epochs = getattr(args, "finetune_epochs", Config.FINETUNE_EPOCHS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(2, epochs // 3), T_mult=1, eta_min=1e-6)

    mixup_alpha = getattr(args, "mixup_alpha", Config.MIXUP_ALPHA)
    grad_clip = Config.GRAD_CLIP
    early_stop = getattr(args, "early_stop", Config.EARLY_STOP)
    metric = getattr(args, "metric", Config.METRIC)

    best = -1.0
    best_state = None
    bad_epochs = 0
    history: List[Dict] = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for input_ids, mask, y in train_loader:
            input_ids, mask, y = input_ids.to(device), mask.to(device), y.to(device)
            logits = model(input_ids, mask)

            # MixUp 在 logits 空间对类别做混合（嵌入空间混合需两次前向，代价太高）
            if mixup_alpha and mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                lam = max(lam, 1.0 - lam)
                idx = torch.randperm(y.size(0), device=device)
                loss = (lam * criterion(logits, y)
                        + (1 - lam) * criterion(logits, y[idx]))
            else:
                loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip)
            optimizer.step()
            total_loss += float(loss) * y.size(0)
            seen += y.size(0)

        scheduler.step(epoch - 1)
        msg = f"[finetune] epoch {epoch:3d}/{epochs}  loss {total_loss / max(seen, 1):.4f}"

        val_metric = None
        if val_loader is not None:
            res = evaluate(model, val_loader, device, num_labels)
            val_metric = res[metric]
            msg += f"  val_acc {res['accuracy']:.4f}  val_{metric} {val_metric:.4f}"
            history.append({"epoch": epoch, "loss": total_loss / max(seen, 1),
                            "val_accuracy": res["accuracy"],
                            f"val_{metric}": val_metric})
            if val_metric > best:
                best = val_metric
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
        else:
            history.append({"epoch": epoch, "loss": total_loss / max(seen, 1)})
        print(msg)

        if val_loader is not None and bad_epochs >= early_stop:
            print(f"[finetune] 早停于 epoch {epoch}（{metric} 连续 {bad_epochs} 轮未提升）")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[finetune] 恢复最佳权重（val_{metric}={best:.4f}）")

    # ── 保存 ──
    if Config.MODEL_PATH is None:
        raise ValueError("MODEL_PATH 未设置")
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    torch.save({
        "state_dict": model.head.state_dict(),
        "num_labels": num_labels,
        "hidden_dim": getattr(args, "hidden_dim", Config.HIDDEN_DIM),
        "input_dim": model.encoder.hidden_size,
        "id2label": id2label or {},
        "encoder": Config.MODEL_NAME,
    }, Config.MODEL_PATH)
    print(f"[finetune] 分类头 → {Config.MODEL_PATH}")

    lora_path = os.path.join(os.path.dirname(Config.MODEL_PATH), "lora_adapter.pt")
    torch.save(lora_state_dict(model.encoder.bert), lora_path)
    print(f"[finetune] LoRA 适配器 → {lora_path} "
          f"（{os.path.getsize(lora_path) / 1e6:.2f} MB）")

    result = {"best_val": best, "metric": metric, "history": history,
              "trainable_params": n_trainable}
    if val_loader is not None:
        final = evaluate(model, val_loader, device, num_labels)
        result["val_accuracy"] = final["accuracy"]
        result["val_macro_f1"] = final["macro_f1"]
        print(f"[finetune] 最终 val_acc {final['accuracy']:.4f} "
              f"val_macro_f1 {final['macro_f1']:.4f}")
    result["elapsed_sec"] = time.time() - t_start
    return result
