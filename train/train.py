"""训练流水线（v2.0：XLM-R 基座 + e2e/cached 双模）。

阶段：clean -> split -> aug -> [embed(仅 cached)] -> train -> cv -> eval -> onnx -> infer

v2 新增：
  * stage_train_e2e   —— 文本级端到端训练（LoRA / 全量），不使用 embeddings.npz
  * stage_calibrate   —— temperature + per-class 阈值
  * stage_onnx        —— 改为导出 encoder+head 整图（v1 只导分类头）
  * cross_validate    —— 支持 e2e
"""

import copy
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
#导入 F
import torch.nn.functional as F

from .augment import EDAAugmentor  # noqa: E402
from .config import Config  # noqa: E402
from .models import (  # noqa: E402
    ChunkedEncoder,
    FocalLoss,
    LightEmotionClassifier,
    ONNXEncoder,
    PoolingStrategy,
    PromptMaskEncoder,
    TextClassificationDataset,
    XlmrTorchEncoder,
    build_encoder_lora,
    build_param_groups,
    load_tokenizer,
    make_collate,
    merge_lora,
)
from .utils import class_weights, macro_f1, mixup, set_seed  # noqa: E402

augmentor = EDAAugmentor()


# ────────────────────────── AMP 兼容层 ──────────────────────────
def _amp_ctx(device_type: str, enabled: bool):
    try:  # torch >= 2.4
        from torch.amp import autocast
        return autocast(device_type=device_type, dtype=torch.float16, enabled=enabled)
    except (ImportError, TypeError):  # pragma: no cover
        from torch.cuda.amp import autocast
        return autocast(enabled=enabled)


def _make_scaler(device_type: str, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.amp import GradScaler
        return GradScaler(device_type)
    except (ImportError, TypeError):  # pragma: no cover
        from torch.cuda.amp import GradScaler
        return GradScaler()


# ────────────────────────── 清洗 ──────────────────────────
class TextCleaner:
    _RE_ZERO_WIDTH = re.compile(r'[\u200b-\u200d\u2028-\u202f\ufeff]')
    _RE_WHITESPACE = re.compile(r'[\u3000]+')
    _RE_CTRL = re.compile(r'[\x00-\x1f\x7f]')
    _RE_HTML = re.compile(r'<[^>]+>')
    _RE_URL = re.compile(r'https?://\S+|www\.\S+')
    _RE_DECO = re.compile(r'[【】■◆▶☞➤]+')
    _RE_SPACE = re.compile(r'[ \t]+')

    @classmethod
    def clean(cls, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
        for r in [cls._RE_CTRL, cls._RE_ZERO_WIDTH, cls._RE_WHITESPACE,
                  cls._RE_HTML, cls._RE_URL, cls._RE_DECO]:
            text = r.sub("", text)
        return cls._RE_SPACE.sub(" ", text).strip()


def load_raw(path: Optional[str] = None) -> pd.DataFrame:
    path = path or Config.DATA_PATH
    for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            raise
    raise FileNotFoundError(path)


def stage_clean(args, df_raw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df_raw is None:
        df_raw = load_raw()

    if "text" not in df_raw.columns or "label" not in df_raw.columns:
        raise KeyError("CSV 必须包含 text 和 label 两列")

    df = df_raw.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).apply(TextCleaner.clean)
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    conflict = df.groupby("text")["label"].nunique()
    bad_texts = set(conflict[conflict > 1].index)
    if bad_texts:
        before = len(df)
        df = df[~df["text"].isin(bad_texts)].reset_index(drop=True)
        print(f"[clean] 剔除标注冲突样本 {before - len(df)} 条（同文本不同标签）")

    lang = getattr(args, "lang", None)
    if lang and "lang" in df.columns:
        keep = [l.strip().lower() for l in lang.split(",") if l.strip()]
        before = len(df)
        df = df[df["lang"].str.lower().isin(keep)].reset_index(drop=True)
        print(f"[clean] 语言过滤 {keep} 后: {len(df)}（原 {before}）")

    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)  # type: ignore
    df.to_csv(os.path.join(Config.OUTPUT_PATH, "clean_data.csv"),  # type: ignore
              index=False, encoding="utf-8-sig")
    print(f"[clean] 清洗后样本数: {len(df)}｜类别数: {df['label'].nunique()}")
    return df


# ────────────────────────── 划分 ──────────────────────────
def _stratify_labels(df: pd.DataFrame, args) -> Optional[pd.Series]:
    if getattr(args, "stratify_lang", False) and "lang" in df.columns:
        return df["label"].astype(str) + "|" + df["lang"].astype(str)
    return df["label"]


def _group_key(df: pd.DataFrame) -> Optional[pd.Series]:
    """跨语言扩充后：同一原文的多语译文必须整组划分，避免 train/test 泄漏。"""
    if "group_id" in df.columns:
        return df["group_id"].astype(str)
    return None


def stage_split(args, df: Optional[pd.DataFrame] = None, folds: int = 1) -> dict:
    if df is None:
        df = pd.read_csv(os.path.join(Config.OUTPUT_PATH, "clean_data.csv"))  # type: ignore

    le = LabelEncoder()
    df = df.copy()
    df["label_id"] = le.fit_transform(df["label"])
    num_labels = len(le.classes_)
    label2id = {c: int(i) for i, c in enumerate(le.classes_)}
    id2label = {int(i): c for i, c in enumerate(le.classes_)}

    strat = _stratify_labels(df, args)
    strat = strat if strat.value_counts().min() >= 2 else None

    train_df, temp_df = train_test_split(
        df, test_size=args.test_size + args.val_size,
        random_state=Config.RANDOM_STATE, stratify=strat,
    )
    val_frac = args.val_size / (args.test_size + args.val_size)
    strat2 = _stratify_labels(temp_df, args)
    strat2 = strat2 if strat2.value_counts().min() >= 2 else None
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_frac,
        random_state=Config.RANDOM_STATE, stratify=strat2,
    )

    # 泄漏防护：把与 train 同 group 的 val/test 样本挪回 train
    gk = _group_key(df)
    if gk is not None:
        df = df.assign(_g=gk)
        train_groups = set(df.loc[df.index.isin(train_df.index), "_g"])
        leak_val = val_df.index.map(lambda i: df.loc[i, "_g"] in train_groups)
        leak_test = test_df.index.map(lambda i: df.loc[i, "_g"] in train_groups)
        if leak_val.any() or leak_test.any():
            print(f"[split] group 泄漏修正：val -{int(np.sum(leak_val))} / test -{int(np.sum(leak_test))}")
            val_df = val_df[~np.asarray(leak_val)]
            test_df = test_df[~np.asarray(leak_test)]

    print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)} labels={num_labels}")

    if Config.WRITE_FILE:
        for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
            cols = ["text", "label", "label_id"] + \
                   [c for c in ["lang", "group_id", "is_mt"] if c in d.columns]
            d[cols].to_csv(os.path.join(Config.OUTPUT_PATH, f"{name}.csv"),  # type: ignore
                           index=False, encoding="utf-8-sig")
        with open(Config.LABELMAP_PATH, "w", encoding="utf-8") as f:  # type: ignore
            json.dump({"label2id": label2id, "id2label": id2label,
                       "num_labels": num_labels}, f, ensure_ascii=False, indent=2)

    return {"train": train_df, "val": val_df, "test": test_df,
            "label2id": label2id, "id2label": id2label, "num_labels": num_labels}


# ────────────────────────── 增强（仅作用于 train）──────────────────────────
def oversample_minority(
    df: pd.DataFrame,
    aug_factor: int = 2,
    minority_threshold: float = 0.8
) -> pd.DataFrame:
    """仅对少数类做过采样。

    v2：EDA 按语言降级 —— 只有 en 走字符级 delete/swap/insert，
    zh/ja 只做安全的 emoji / 前后缀，避免破坏语法与情感标签。
    """
    counts = df["label"].value_counts()
    mean_n = counts.mean()
    thr = minority_threshold * mean_n
    new_rows = []
    for _, row in df.iterrows():
        new_rows.append(row)
        if counts[row["label"]] < thr:
            lang = str(row.get("lang", "en")).lower()
            for _ in range(aug_factor):
                aug = row.copy()
                aug["text"] = augmentor.eda_combine(row["text"], label=row["label"], lang=lang)
                if aug["text"] != row["text"]:
                    new_rows.append(aug)
    out = pd.DataFrame(new_rows).drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return out


def stage_augment(args, split: dict) -> dict:
    if not getattr(args, "use_aug", True):
        print("[aug] 已关闭增强")
        return split

    train_df = split["train"]
    before = len(train_df)
    train_df = oversample_minority(
        train_df,
        aug_factor=getattr(args, "aug_factor", Config.AUG_FACTOR),
        minority_threshold=Config.MINORITY_THRESHOLD,
    ).reset_index(drop=True)

    # 回译（zh/en/ja 往返 pivot）
    if getattr(args, "use_bt", Config.USE_BACKTRANS):
        from .backtranslate import augment_with_backtranslation
        train_df = augment_with_backtranslation(
            train_df, factor=getattr(args, "bt_factor", Config.BT_FACTOR),
            model_name=getattr(args, "bt_model", Config.BT_MODEL),
            cache_path=Config.BT_CACHE_PATH, device=args.device,
        ).reset_index(drop=True)

    split["train"] = train_df
    print(f"[aug] 训练集: {before} → {len(train_df)}（仅 train，val/test 保持原样）")

    if Config.WRITE_FILE:
        cols = ["text", "label", "label_id"] + \
               [c for c in ["lang", "group_id", "is_mt"] if c in train_df.columns]
        train_df[cols].to_csv(os.path.join(Config.OUTPUT_PATH, "train_aug.csv"),  # type: ignore
                              index=False, encoding="utf-8-sig")
    return split


# ────────────────────────── 编码（cached 模式）──────────────────────────
def ensure_encoder_dir() -> str:
    """确保训练基座目录就绪。

    优先使用三语裁剪产物（Config.base_model_dir()）；
    不存在时从 hub 下载原始 XLM-R 并缓存到 models/<hub_name>。

    ⚠️ 这里返回的是 **HuggingFace 格式目录**（config.json + 权重 + spm），
    不是 ONNX —— 训练必须加载 HF 权重，ONNX 只在推理部署侧出现。
    """
    local = Config.base_model_dir()
    if Config.is_complete_hf_dir(local):
        print(f"[model] 使用三语裁剪基座: {local}")
        return local

    if not Config.FALLBACK_TO_HUB:
        raise FileNotFoundError(
            f"基座目录不完整: {local}\n"
            f"请先: python scripts/vocab/trim_vocab.py --mode intersect --out {local}")

    if not Config.MODEL_AUTO_DOWNLOAD:
        raise FileNotFoundError(f"基座目录不存在: {local}")

    cache = Path(Config.MODEL_CACHE_DIR)  # type: ignore
    model_path = cache / Config.HUB_MODEL_NAME.replace("/", "_")
    if Config.is_complete_hf_dir(str(model_path)):
        print(f"[model] 使用缓存的原始基座: {model_path}")
        return str(model_path)

    print(f"[model] 首次运行，正在下载 {Config.HUB_MODEL_NAME} (~1.1GB)...")
    from transformers import XLMRobertaModel
    m = XLMRobertaModel.from_pretrained(Config.HUB_MODEL_NAME)
    m.save_pretrained(str(model_path))
    load_tokenizer(Config.HUB_MODEL_NAME).save_pretrained(str(model_path))
    print(f"[model] 下载完成 → {model_path}")
    return str(model_path)


def _load_onnx_encoder(onnx_path: str, tokenizer_dir: Optional[str] = None) -> ONNXEncoder:
    return ONNXEncoder(onnx_path=onnx_path, tokenizer_dir=tokenizer_dir,
                       tokenizer_name=Config.MODEL_NAME)


def stage_embed(args, split: Optional[dict] = None, backend: str = "auto") -> dict:
    """预编码缓存（仅 cached / prompt 路径使用）。

    v2：编码器由 SentenceTransformer(MiniLM) 换成 XLM-R；
    USE_PROMPT=True 时取 [MASK] 位置的 hidden state 而不是句向量。
    """
    if split is None:
        split = {k: pd.read_csv(os.path.join(Config.OUTPUT_PATH, f"{k}.csv"))  # type: ignore
                 for k in ["train", "val", "test"]}

    if backend == "auto":
        backend = "onnx" if os.path.exists(Config.onnx_encoder_cache()) else "pytorch"

    use_prompt = getattr(args, "use_prompt", Config.USE_PROMPT)

    if backend == "onnx":
        print(f"[embed] 使用 ONNX 编码器缓存: {Config.onnx_encoder_cache()}")
        encoder = _load_onnx_encoder(Config.onnx_encoder_cache(),
                                     Config.tokenizer_dir_for_train())
        def encode_list(texts, langs=None):
            return encoder.encode(texts, lang=langs, batch_size=args.batch_size, normalize=False)
    else:
        device_str = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
        print(f"[embed] using device: {device_str}｜prompt={use_prompt}")
        if use_prompt:
            encoder = PromptMaskEncoder(device=device_str)
        else:
            encoder = XlmrTorchEncoder(device=device_str)

        def encode_list(texts, langs=None):
            return encoder.encode(texts, lang=langs, batch_size=args.batch_size, normalize=False)

    result: Dict[str, np.ndarray] = {}
    for name in ["train", "val", "test"]:
        texts = split[name]["text"].astype(str).tolist()
        langs = split[name]["lang"].tolist() if "lang" in split[name].columns else None
        print(f"[embed] 编码 {name} ({len(texts)} 条)...")
        result[f"X_{name}"] = encode_list(texts, langs).astype(np.float32)
        result[f"y_{name}"] = split[name]["label_id"].values.astype(np.int64)
        if "is_mt" in split[name].columns:
            result[f"w_{name}"] = np.where(split[name]["is_mt"].values == 1,
                                           Config.CL_SAMPLE_WEIGHT, 1.0).astype(np.float32)

    np.savez_compressed(Config.EMBED_PATH, **result)  # type: ignore
    print(f"[embed] 已保存 → {Config.EMBED_PATH}")  # type: ignore
    return result


# ────────────────────────── 训练：cached（只训分类头）──────────────────────────
def _build_loaders(emb: dict, args, num_workers: int = 0):
    loaders = {}
    for name in ["train", "val", "test"]:
        tensors = [torch.from_numpy(np.asarray(emb[f"X_{name}"], dtype=np.float32)),
                   torch.from_numpy(np.asarray(emb[f"y_{name}"], dtype=np.int64))]
        if f"w_{name}" in emb:
            tensors.append(torch.from_numpy(np.asarray(emb[f"w_{name}"], dtype=np.float32)))
        ds = TensorDataset(*tensors)
        loaders[name] = DataLoader(
            ds, batch_size=args.batch_size, shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=(args.device == "cuda" and torch.cuda.is_available()),
            persistent_workers=False,
            drop_last=(name == "train" and len(ds) % args.batch_size == 1),
        )
    return loaders


def update_ema(model, ema_model, decay):
    with torch.no_grad():
        for p, ema_p in zip(model.parameters(), ema_model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def _ema_decay(epoch: int, base: float, warmup: int) -> float:
    if warmup <= 0 or epoch >= warmup:
        return base
    return min(base, (1 + epoch) / (warmup + epoch))


def _metric(y_true, y_pred, num_labels: int, name: str) -> float:
    if name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def stage_train(args, emb: Optional[dict] = None, num_labels: Optional[int] = None):
    """cached 模式：冻结编码器 + 预编码向量 -> 只训分类头（沿用 v1）。"""
    if emb is None:
        emb = dict(np.load(Config.EMBED_PATH))  # type: ignore
    if num_labels is None:
        with open(Config.LABELMAP_PATH, encoding="utf-8") as f:  # type: ignore
            num_labels = json.load(f)["num_labels"]

    set_seed(getattr(args, "seed", Config.SEED))
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    loaders = _build_loaders(emb, args, num_workers=getattr(args, "num_workers", Config.NUM_WORKERS))

    input_dim = int(np.asarray(emb["X_train"]).shape[1])
    model = LightEmotionClassifier(
        input_dim=input_dim, num_classes=num_labels,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        stochastic_p=Config.STOCHASTIC_P,
    ).to(device)

    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    ema_model = copy.deepcopy(model)

    counts = np.bincount(np.asarray(emb["y_train"]).astype(int), minlength=num_labels)
    weights = torch.tensor(class_weights(counts, Config.CLASS_WEIGHT,
                                         Config.EFFECTIVE_BETA), dtype=torch.float).to(device)
    criterion = FocalLoss(gamma=Config.FOCAL_GAMMA if Config.USE_FOCAL_LOSS else 0.0,
                          weight=weights, label_smoothing=Config.LABEL_SMOOTHING).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=getattr(args, "lr", Config.CACHED_LR),
                                  weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.SCHED_T0, T_mult=Config.SCHED_T_MULT, eta_min=Config.SCHED_ETA_MIN)

    use_amp = device.type == "cuda"
    scaler = _make_scaler(device.type, use_amp)

    best_score, bad_epochs, history = -1.0, 0, []
    metric_name = getattr(args, "metric", Config.METRIC)
    epochs = getattr(args, "epochs", Config.CACHED_EPOCHS)
    early_stop = getattr(args, "early_stop", Config.CACHED_EARLY_STOP)

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for batch in loaders["train"]:
            xb, yb = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            sw = batch[2].to(device) if len(batch) > 2 else None
            optimizer.zero_grad(set_to_none=True)
            mixed_xb, ya, yb2, lam = mixup(xb, yb, alpha=Config.MIXUP_ALPHA)
            if use_amp:
                with _amp_ctx(device.type, True):
                    logits, _, _ = model(mixed_xb)
                    loss = lam * criterion(logits, ya, sw) + (1 - lam) * criterion(logits, yb2, sw)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _, _ = model(mixed_xb)
                loss = lam * criterion(logits, ya, sw) + (1 - lam) * criterion(logits, yb2, sw)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
                optimizer.step()

            update_ema(model, ema_model, _ema_decay(epoch, Config.EMA_DECAY, Config.EMA_WARMUP_EPOCHS))
            tot_loss += loss.item() * len(yb)
            correct += (logits.float().argmax(1) == yb).sum().item()
            total += len(yb)

        scheduler.step(epoch - 1)

        ema_model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in loaders["val"]:
                xb = batch[0].to(device, non_blocking=True)
                pred = ema_model(xb)[0].argmax(1).cpu().numpy()
                y_pred.extend(pred.tolist())
                y_true.extend(batch[1].numpy().tolist())
        score = _metric(y_true, y_pred, num_labels, metric_name)
        acc = float(accuracy_score(y_true, y_pred))
        history.append((epoch, tot_loss / max(1, total), correct / max(1, total), score, acc))
        print(f"[train] ep {epoch:3d} | loss {history[-1][1]:.4f} | train {history[-1][2]:.4f} "
              f"| {metric_name} {score:.4f} | acc {acc:.4f} "
              f"| lr {optimizer.param_groups[0]['lr']:.2e}")

        if score > best_score + 1e-5:
            best_score, bad_epochs = score, 0
            torch.save({"state_dict": ema_model.state_dict(), "num_labels": num_labels,
                        "hidden_dim": args.hidden_dim, "input_dim": input_dim,
                        "id2label": json.load(open(Config.LABELMAP_PATH,  # type: ignore
                                                   encoding="utf-8"))["id2label"]},
                       Config.MODEL_PATH)  # type: ignore
            print(f"[train] 保存最佳模型 @ ep {epoch}（{metric_name}={score:.4f}）")
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop:
                print(f"[train] 早停 @ ep {epoch} (best {metric_name}={best_score:.4f})")
                break

    print(f"[train] 最佳 {metric_name}: {best_score:.4f} → {Config.MODEL_PATH}")  # type: ignore
    if Config.WRITE_FILE:
        with open(os.path.join(Config.OUTPUT_PATH, "train_history.json"),  # type: ignore
                  "w", encoding="utf-8") as f:
            json.dump({"metric": metric_name, "best": best_score,
                       "history": [{"epoch": h[0], "loss": h[1], "train_acc": h[2],
                                    metric_name: h[3], "val_acc": h[4]} for h in history]},
                      f, ensure_ascii=False, indent=2)
    return model


# ────────────────────────── 训练：e2e（LoRA）──────────────────────────
def _fgm_attack(model, eps: float):
    """Fast Gradient Method：在 embedding 上加扰动（只对 e2e 的 embedding 生效）。"""
    emb = None
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            emb = m
            break
    if emb is None or emb.weight.grad is None:
        return None
    norm = torch.norm(emb.weight.grad)
    if norm < 1e-12:
        return None
    r_at = eps * emb.weight.grad / norm
    emb.weight.data.add_(r_at)
    return (emb, r_at)


def _rdrop_kl(logits1, logits2, alpha: float) -> torch.Tensor:
    p1 = F.log_softmax(logits1, dim=-1)
    p2 = F.log_softmax(logits2, dim=-1)
    return alpha * 0.5 * (F.kl_div(p1, p2, reduction="batchmean", log_target=True)
                          + F.kl_div(p2, p1, reduction="batchmean", log_target=True))


def stage_train_e2e(args, split: dict, num_labels: int):
    """端到端训练（路径 A：XLM-R + LoRA）。

    * 不使用 embeddings.npz（编码器参与训练，缓存必然失效）；
    * 支持 R-Drop / FGM / EMA / LLRD / 分组 lr；
    * 训练结束 merge_and_unload，返回标准 XLM-R 结构，供剪枝/导出使用。
    """
    set_seed(getattr(args, "seed", Config.SEED))
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    tokenizer = load_tokenizer(Config.encoder_dir(),
                               Config.tokenizer_dir_for_train())
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    use_lang_token = getattr(args, "use_lang_token", Config.USE_LANG_TOKEN)
    model = build_encoder_lora(
        num_labels=num_labels,
        model_name=getattr(args, "model_name", Config.MODEL_NAME),
        r=getattr(args, "lora_r", Config.LORA_R),
        unfreeze_from=getattr(args,"lora_unfreeze_from",Config.LORA_UNFREEZE_FROM)
    )
    if use_lang_token:
        _add_lang_tokens(model, tokenizer)
    model.to(device)

    def _mk(name: str, shuffle: bool):
        d = split[name]
        langs = d["lang"].tolist() if "lang" in d.columns else None
        sw = None
        if "is_mt" in d.columns:
            sw = np.where(d["is_mt"].values == 1, Config.CL_SAMPLE_WEIGHT, 1.0).astype(np.float32)
        ds = TextClassificationDataset(
            d["text"].astype(str).tolist(),
            d["label_id"].values, 
            tokenizer,
            max_len=getattr(args, "max_seq_len", Config.MAX_SEQ_LEN),
            lang=langs, 
            use_lang_token=use_lang_token,                           
            sample_weight=sw
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=getattr(args, "num_workers", Config.NUM_WORKERS),
                          collate_fn=make_collate(pad_id=pad_id),
                          pin_memory=(device.type == "cuda"),
                          drop_last=(shuffle and len(ds) % args.batch_size == 1))

    loaders = {n: _mk(n, n == "train") for n in ["train", "val", "test"]}

    counts = np.bincount(np.asarray(split["train"]["label_id"]).astype(int), minlength=num_labels)
    weights = torch.tensor(class_weights(counts, Config.CLASS_WEIGHT,
                                         Config.EFFECTIVE_BETA), dtype=torch.float).to(device)
    criterion = FocalLoss(gamma=Config.FOCAL_GAMMA if Config.USE_FOCAL_LOSS else 0.0,
                          weight=weights, label_smoothing=Config.LABEL_SMOOTHING).to(device)

    groups = build_param_groups(model, lora_lr=getattr(args, "lr", Config.LORA_LR),
                                head_lr=Config.HEAD_LR, llrd=Config.LLRD)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.SCHED_T0, T_mult=Config.SCHED_T_MULT, eta_min=Config.SCHED_ETA_MIN)

    use_amp = device.type == "cuda"
    scaler = _make_scaler(device.type, use_amp)
    use_ema = getattr(args, "use_ema", True)
    ema_model = copy.deepcopy(model).to(device) if use_ema else None

    best_score, bad_epochs, history = -1.0, 0, []
    metric_name = getattr(args, "metric", Config.METRIC)
    epochs = getattr(args, "epochs", Config.EPOCHS)
    early_stop = getattr(args, "early_stop", Config.EARLY_STOP)

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for batch in loaders["train"]:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            yb = batch["labels"].to(device, non_blocking=True)
            sw = batch.get("sample_weight")
            sw = sw.to(device) if sw is not None else None

            def _step():
                out = model(input_ids=ids, attention_mask=mask)
                logits = out.logits
                loss = criterion(logits, yb, sw)
                if Config.R_DROP_ALPHA > 0:
                  out2 = model(input_ids=ids, attention_mask=mask)
                  loss = loss + _rdrop_kl(logits, out2.logits, Config.R_DROP_ALPHA)
                  return logits, loss   # 注意：不在这里 backward

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with _amp_ctx(device.type, True):
                    logits, loss = _step()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if Config.FGM_EPS > 0:
                    backup = _fgm_attack(model, Config.FGM_EPS)
                    if backup is not None:
                        with _amp_ctx(device.type, True):
                            _, adv_loss = _step()
                        adv_loss.backward()
                        emb, r_at = backup
                        emb.weight.data.sub_(r_at)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss = _step()
                loss.backward()
                if Config.FGM_EPS > 0:
                    backup = _fgm_attack(model, Config.FGM_EPS)
                    if backup is not None:
                        _, adv_loss = _step()
                        adv_loss.backward()
                        emb, r_at = backup
                        emb.weight.data.sub_(r_at)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
                optimizer.step()

            if ema_model is not None:
                update_ema(model, ema_model, 
                    _ema_decay(epoch, Config.EMA_DECAY,
                                                        Config.EMA_WARMUP_EPOCHS)
                )
            tot_loss += loss.item() * len(yb)
            correct += (logits.float().argmax(1) == yb).sum().item()
            total += len(yb)

        scheduler.step(epoch - 1)

        eval_model = ema_model or model
        eval_model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in loaders["val"]:
                out = eval_model(input_ids=batch["input_ids"].to(device),
                                 attention_mask=batch["attention_mask"].to(device))
                y_pred.extend(out.logits.argmax(1).cpu().numpy().tolist())
                y_true.extend(batch["labels"].numpy().tolist())
        score = _metric(y_true, y_pred, num_labels, metric_name)
        acc = float(accuracy_score(y_true, y_pred))
        history.append((epoch, tot_loss / max(1, total), correct / max(1, total), score, acc))
        print(f"[e2e] ep {epoch:3d} | loss {history[-1][1]:.4f} | train {history[-1][2]:.4f} "
              f"| {metric_name} {score:.4f} | acc {acc:.4f} "
              f"| lr {optimizer.param_groups[0]['lr']:.2e}")

        if score > best_score + 1e-5:
            best_score, bad_epochs = score, 0
            merged = merge_lora(copy.deepcopy(eval_model))
            torch.save({"state_dict": merged.state_dict(), "num_labels": num_labels,
                        "model_name": Config.encoder_dir(),
                        "encoder_dir": Config.encoder_dir(),
                        "trimmed": Config.is_trimmed(), "train_mode": "e2e",
                        "id2label": json.load(open(Config.LABELMAP_PATH,  # type: ignore
                                                   encoding="utf-8"))["id2label"]},
                       Config.MODEL_PATH)  # type: ignore
            print(f"[e2e] 保存最佳模型 @ ep {epoch}（{metric_name}={score:.4f}）")
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop:
                print(f"[e2e] 早停 @ ep {epoch} (best {metric_name}={best_score:.4f})")
                break

    print(f"[e2e] 最佳 {metric_name}: {best_score:.4f} → {Config.MODEL_PATH}")  # type: ignore
    if Config.WRITE_FILE:
        with open(os.path.join(Config.OUTPUT_PATH, "train_history.json"),  # type: ignore
                  "w", encoding="utf-8") as f:
            json.dump({"metric": metric_name, "best": best_score, "mode": "e2e",
                       "history": [{"epoch": h[0], "loss": h[1], "train_acc": h[2],
                                    metric_name: h[3], "val_acc": h[4]} for h in history]},
                      f, ensure_ascii=False, indent=2)
    return merge_lora(model)


def _get_encoder_module(model):
    """取出真正持有 word_embeddings 的编码器模块。"""
    # 原生 XLMRobertaForSequenceClassification：model.roberta
    if hasattr(model, "roberta"):
        return model.roberta
    # PEFT 包装：model.base_model.model.roberta
    if hasattr(model, "base_model"):
        inner = model.base_model
        if hasattr(inner, "model") and hasattr(inner.model, "roberta"):
            return inner.model.roberta
        if hasattr(inner, "roberta"):
            return inner.roberta
    # 兜底：直接用 get_input_embeddings（大多数 HF 模型和 PEFT 模型都提供）
    return model

def _add_lang_tokens(model, tokenizer):
    new_tokens = [t for t in Config.LANG_TOKENS.values() if t not in tokenizer.get_vocab()]
    if not new_tokens:
        return
    tokenizer.add_tokens(new_tokens, special_tokens=True)

    base = _get_encoder_module(model)
    base.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    with torch.no_grad():
        emb = base.get_input_embeddings().weight
        anchor = emb[tokenizer.mask_token_id or tokenizer.unk_token_id].detach().clone()
        for t in new_tokens:
            emb[tokenizer.convert_tokens_to_ids(t)] = anchor


# ────────────────────────── 校准 ──────────────────────────
def stage_calibrate(
    args, split: Optional[dict] = None,
    logits: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None
) -> dict:
    """Temperature scaling（单标签）+ 可选 per-class 阈值。

    产物 calib.json：{"temperature": T, "per_class": {...}, "per_lang": {...}}
    """
    import scipy.optimize as opt  # 延迟导入

    ckpt_path = Config.CALIB_PATH  # type: ignore
    if logits is None:
        with open(os.path.join(Config.OUTPUT_PATH, "val_logits.npz"), "rb") as f:  # type: ignore
            z = np.load(f)
            logits, y_true = z["logits"], z["labels"]
    assert logits is not None and y_true is not None

    def nll(T):
        T = max(float(T[0]), 1e-3)
        p = logits / T
        p = p - p.max(axis=1, keepdims=True)
        e = np.exp(np.clip(p, -60, None))
        e = e / e.sum(axis=1, keepdims=True)
        return float(-np.log(np.clip(e[np.arange(len(y_true)), y_true], 1e-12, None)).mean())

    res = opt.minimize(nll, x0=[1.0], bounds=[(0.05, 10.0)], method="L-Bounds" if False else "Powell")
    temperature = float(res.x[0])

    calib = {
        "temperature": round(temperature, 4),
        "per_class": {},
        "per_lang": {}
    }
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print(f"[calib] temperature={temperature:.4f} → {ckpt_path}")
    return calib


# ────────────────────────── 评估 ──────────────────────────
def stage_eval(
    args,
    emb: Optional[dict] = None,
    id2label: Optional[dict] = None,
    split: Optional[dict] = None
):
    """评估。cached 用向量；e2e 传 split 走文本前向。"""
    if id2label is None:
        with open(Config.LABELMAP_PATH, encoding="utf-8") as f:
            id2label = json.load(f)["id2label"]
    num_labels = len(id2label)

    device = torch.device("cpu")
    ckpt = torch.load(Config.MODEL_PATH, map_location=device)
    state = ckpt.get("state_dict", ckpt)

    # 关键：用 state_dict 的键判断，不依赖 ckpt 元数据或 args
    is_e2e = any(k.startswith("roberta.") for k in state.keys())
    print(f"[eval] {Config.MODEL_PATH} | is_e2e={is_e2e} | "
          f"train_mode={ckpt.get('train_mode')} | num_labels={num_labels}")

    if is_e2e:
        from transformers import XLMRobertaForSequenceClassification
        encoder_dir = ckpt.get("encoder_dir") or Config.encoder_dir()
        model = XLMRobertaForSequenceClassification.from_pretrained(
            encoder_dir, num_labels=num_labels)
        # vocab 对齐（加了 lang token 时会差 3）
        ckpt_vocab = state["roberta.embeddings.word_embeddings.weight"].shape[0]
        if ckpt_vocab != model.config.vocab_size:
            model.resize_token_embeddings(ckpt_vocab, mean_resizing=False)
        model.load_state_dict(state)
        model.to(device).eval()
        tokenizer = load_tokenizer(encoder_dir)
        # 若 ckpt_vocab 比 base 多 3，说明存的是 lang token 版本
        base_vocab = model.config.vocab_size - len(Config.LANG_TOKENS)
        if ckpt_vocab > base_vocab:
            _add_lang_tokens(model, tokenizer)
        text_pred = True
    else:
        hidden_dim = ckpt.get("hidden_dim", Config.HIDDEN_DIM)
        input_dim = ckpt.get("input_dim", Config.INPUT_DIM)
        model = LightEmotionClassifier(input_dim=input_dim, num_classes=num_labels,
                                       hidden_dim=hidden_dim, dropout=0.0).to(device)
        model.load_state_dict(state)
        model.eval()
        text_pred = False

    if text_pred:
        assert split is not None, "e2e 评估需要 split"
        d = split["test"]
        texts = d["text"].astype(str).tolist()
        langs = d["lang"].tolist() if "lang" in d.columns else None
        probs_all = []
        bs = getattr(args, "batch_size", 32)
        with torch.no_grad():
            for s in range(0, len(texts), bs):
                chunk = texts[s:s + bs]
                if Config.USE_LANG_TOKEN and langs is not None:
                    chunk = [f"{Config.LANG_TOKENS.get(str(langs[s + i]).lower(), '')}{t}"
                             for i, t in enumerate(chunk)]
                enc = tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=getattr(args, "max_seq_len", Config.MAX_SEQ_LEN),
                    return_tensors="pt"
                )
                logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
                probs_all.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        probs = np.concatenate(probs_all, axis=0)
        y_test = np.asarray(d["label_id"]).astype(int)
    else:
        if emb is None:
            emb = dict(np.load(Config.EMBED_PATH))  # type: ignore
        X_test = torch.from_numpy(np.asarray(emb["X_test"], dtype=np.float32))
        y_test = np.asarray(emb["y_test"]).astype(int)
        with torch.no_grad():
            probs = torch.softmax(model(X_test)[0], dim=1).numpy()

    preds = probs.argmax(1)
    acc = accuracy_score(y_test, preds)
    mf1 = f1_score(y_test, preds, average="macro", zero_division=0)
    print(f"\n[eval] Test Accuracy: {acc:.4f} | Macro-F1: {mf1:.4f}")

    id2label_i = {int(k): v for k, v in id2label.items()}
    target_names = [id2label_i[i] for i in range(num_labels)]
    report = classification_report(y_test, preds, target_names=target_names,
                                   digits=4, zero_division=0)
    print(report)

    per_lang = {}
    # ── 为 stage_calibrate 保存 val 原始 logits ──
    if split is not None and "val" in split and Config.CALIBRATE:
        dv = split["val"]
        val_texts = dv["text"].astype(str).tolist()
        val_langs = dv["lang"].tolist() if "lang" in dv.columns else None
        val_logits_all = []
        with torch.no_grad():
            for s in range(0, len(val_texts), bs):
                chunk = val_texts[s:s + bs]
                if Config.USE_LANG_TOKEN and val_langs is not None:
                    chunk = [
                        f"{Config.LANG_TOKENS.get(str(val_langs[s + i]).lower(), '')}{t}"
                        for i, t in enumerate(chunk)
                    ]
                enc = tokenizer(
                    chunk, padding=True, truncation=True,
                    max_length=getattr(args, "max_seq_len", Config.MAX_SEQ_LEN),
                    return_tensors="pt",
                )
                logits_v = model(**{k: v.to(device) for k, v in enc.items()}).logits
                val_logits_all.append(logits_v.float().cpu().numpy())
        np.savez(
            os.path.join(Config.OUTPUT_PATH, "val_logits.npz"),
            logits=np.concatenate(val_logits_all, axis=0),
            labels=np.asarray(dv["label_id"]).astype(int),
        )
    if Config.WRITE_FILE:
        with open(os.path.join(Config.OUTPUT_PATH, "metrics.json"),  # type: ignore
                  "w", encoding="utf-8") as f:
            json.dump({
                "accuracy": float(acc),
                "macro_f1": float(mf1),
                "per_lang": per_lang,
                "report": report
            },f, ensure_ascii=False, indent=2)
    return {
        "accuracy": float(acc),
        "macro_f1": float(mf1),
        "per_lang": per_lang
    }


# ────────────────────────── ONNX 导出（整图）──────────────────────────
def stage_onnx(args, num_labels: Optional[int] = None) -> None:
    """v2：导出 encoder+head 整图（v1 只导 384->128->18 的分类头）。

    输入：input_ids / attention_mask（动态 batch 与 sequence）
    输出：logits
    """
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if num_labels is None:
        with open(Config.LABELMAP_PATH, encoding="utf-8") as f:  # type: ignore
            num_labels = json.load(f)["num_labels"]

    ckpt = torch.load(Config.MODEL_PATH, map_location="cpu")  # type: ignore
    state = ckpt.get("state_dict", ckpt)
    max_len = getattr(args, "max_seq_len", Config.MAX_SEQ_LEN)

    if ckpt.get("train_mode") == "e2e":
        from transformers import XLMRobertaForSequenceClassification
        model = XLMRobertaForSequenceClassification.from_pretrained(
            ckpt.get("encoder_dir") or Config.encoder_dir(), num_labels=num_labels)
        model.load_state_dict(state)
        model.eval()
        dummy_ids = torch.randint(0, 1000, (1, max_len), dtype=torch.long)
        dummy_mask = torch.ones((1, max_len), dtype=torch.long)
        dummy = (dummy_ids, dummy_mask)
        input_names = ["input_ids", "attention_mask"]
        dynamic_axes = {"input_ids": {0: "batch", 1: "sequence"},
                        "attention_mask": {0: "batch", 1: "sequence"},
                        "logits": {0: "batch"}}
    else:
        hidden_dim = ckpt.get("hidden_dim", Config.HIDDEN_DIM)
        input_dim = ckpt.get("input_dim", Config.INPUT_DIM)
        model = LightEmotionClassifier(input_dim=input_dim, num_classes=num_labels,
                                       hidden_dim=hidden_dim, dropout=0.0).eval()
        model.load_state_dict(state)
        dummy = torch.randn(1, input_dim)
        input_names = ["input"]
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        model, dummy, Config.ONNX_PATH,  # type: ignore
        input_names=input_names, output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,  # 18 在部分 runtime 上不支持 LayerNorm 融合
        do_constant_folding=True,
    )
    try:
        from onnxsim import simplify
        m = onnx.load(Config.ONNX_PATH)  # type: ignore
        m, ok = simplify(m)
        if ok:
            onnx.save(m, Config.ONNX_PATH)  # type: ignore
    except Exception as e:
        print(f"[onnx] onnxsim 跳过（{e}）")
    onnx.checker.check_model(onnx.load(Config.ONNX_PATH))  # type: ignore
    print(f"[onnx] FP32 整图已导出 → {Config.ONNX_PATH}")  # type: ignore

    try:
        from onnxconverter_common import float16
        fp16_model = float16.convert_float_to_float16(
            onnx.load(Config.ONNX_PATH), keep_io_types=True)  # type: ignore
        onnx.save(fp16_model, Config.ONNX_FP16_PATH)  # type: ignore
        print(f"[onnx] FP16 → {Config.ONNX_FP16_PATH}")  # type: ignore
    except Exception as e:
        print(f"[onnx] FP16 导出跳过（{e}）")

    quant_ok = False
    try:
        quantize_dynamic(model_input=Config.ONNX_PATH,  # type: ignore
                         model_output=Config.ONNX_QUANT_PATH,  # type: ignore
                         weight_type=QuantType.QInt8)
        quant_ok = os.path.exists(Config.ONNX_QUANT_PATH)  # type: ignore
        print(f"[onnx] INT8 → {Config.ONNX_QUANT_PATH}")  # type: ignore
    except Exception as e:
        print(f"[onnx] 量化失败（{e}），将使用 FP32 模型进行推理")

    sess = ort.InferenceSession(Config.ONNX_PATH, providers=["CPUExecutionProvider"])  # type: ignore
    ref = None
    for b in [1, 2, 8]:
        feeds = {}
        for i in sess.get_inputs():
            if i.name == "input_ids":
                feeds[i.name] = np.random.randint(0, 1000, (b, max_len)).astype(np.int64)
            elif i.name == "attention_mask":
                feeds[i.name] = np.ones((b, max_len), dtype=np.int64)
            else:
                feeds[i.name] = np.random.randn(b, ckpt.get("input_dim", 768)).astype(np.float32)
        out = sess.run(["logits"], feeds)[0]
        assert out.shape == (b, num_labels), out.shape
        if b == 8:
            ref = out
    if quant_ok and ref is not None:
        qsess = ort.InferenceSession(Config.ONNX_QUANT_PATH,  # type: ignore
                                     providers=["CPUExecutionProvider"])
        feeds = {}
        for i in qsess.get_inputs():
            if i.name == "input_ids":
                feeds[i.name] = np.random.randint(0, 1000, (8, max_len)).astype(np.int64)
            elif i.name == "attention_mask":
                feeds[i.name] = np.ones((8, max_len), dtype=np.int64)
            else:
                feeds[i.name] = np.random.randn(8, ckpt.get("input_dim", 768)).astype(np.float32)
        qout = qsess.run(["logits"], feeds)[0]
        print(f"[onnx] 自检通过；INT8 与 FP32 最大偏差 "
              f"{float(np.abs(qout - ref).max()):.4f}")
    else:
        print(f"[onnx] 自检通过（batch 1/2/8，num_labels={num_labels}）")


# ────────────────────────── 推理 ──────────────────────────
def stage_infer(args, texts: Optional[List[str]] = None) -> None:
    import onnxruntime as ort

    if texts is None:
        texts = list(getattr(args, "infer_text", None) or [])
    if len(texts) == 0:
        return

    with open(Config.LABELMAP_PATH, encoding="utf-8") as f:  # type: ignore
        id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}

    onnx_path = Config.ONNX_QUANT_PATH if os.path.exists(  # type: ignore
        Config.ONNX_QUANT_PATH) else Config.ONNX_PATH  # type: ignore
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])  # type: ignore
    names = [i.name for i in sess.get_inputs()]

    if "input_ids" in names:      # 整图
        tokenizer = load_tokenizer(Config.MODEL_NAME)
        max_len = getattr(args, "max_seq_len", Config.MAX_SEQ_LEN)
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len,
                        return_tensors="np")
        feeds = {}
        for i in sess.get_inputs():
            if i.name in enc:
                feeds[i.name] = enc[i.name].astype(np.int64)
            else:
                feeds[i.name] = np.zeros_like(enc["input_ids"]).astype(np.int64)
        logits = sess.run(["logits"], feeds)[0]
    else:                          # 仅分类头（旧产物）
        encoder = _load_onnx_encoder(Config.onnx_encoder_cache(),
                                     Config.tokenizer_dir_for_train())
        embs = encoder.encode(texts, normalize=False)
        logits = sess.run(["logits"], {"input": embs})[0]

    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(logits, -60, None))
    probs = exp / exp.sum(axis=1, keepdims=True)

    top_k = max(1, min(int(args.top_k), probs.shape[1]))
    for i, t in enumerate(texts):
        top = np.argsort(probs[i])[::-1][:top_k]
        res = ", ".join(f"{id2label.get(int(j), j)}({probs[i][j]:.2f})" for j in top)
        print(f"[infer] 「{t}」 → {res}")


def cross_validate(args, emb: dict, num_labels: int, folds: int = 5,
                   split: Optional[dict] = None) -> None:
    """小数据集上的可靠评估：固定测试集 + 训练集 K 折，报告均值±标准差。"""
    from sklearn.model_selection import StratifiedKFold

    if getattr(args, "train_mode", Config.TRAIN_MODE) == "e2e" and split is not None:
        tr = pd.concat([split["train"], split["val"]], ignore_index=True)
        test_df = split["test"]
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=Config.RANDOM_STATE)
        scores = []
        for k, (a, b) in enumerate(skf.split(tr, tr["label_id"]), 1):
            print(f"\n[cv] fold {k}/{folds}")
            fold_split = {"train": tr.iloc[a].reset_index(drop=True),
                          "val": tr.iloc[b].reset_index(drop=True),
                          "test": test_df}
            stage_train_e2e(args, fold_split, num_labels)
            res = stage_eval(args, None, split["id2label"], fold_split)
            scores.append(res["macro_f1"])
        print(f"\n[cv] macro-F1: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        return

    X = np.concatenate([emb["X_train"], emb["X_val"]])
    y = np.concatenate([emb["y_train"], emb["y_val"]]).astype(int)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=Config.RANDOM_STATE)
    scores = []
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        fold_emb = {"X_train": X[tr], "y_train": y[tr],
                    "X_val": X[va], "y_val": y[va],
                    "X_test": emb["X_test"], "y_test": emb["y_test"]}
        print(f"\n[cv] fold {k}/{folds}")
        stage_train(args, fold_emb, num_labels)
        scores.append(stage_eval(args, fold_emb)["macro_f1"])
    print(f"\n[cv] macro-F1: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
