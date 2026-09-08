import copy
import json
import os
import re
import sys
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

from .augment import EDAAugmentor  # noqa: E402
from .config import Config  # noqa: E402
from .models import (  # noqa: E402
    ChunkedEncoder,
    FocalLoss,
    LightEmotionClassifier,
    ONNXEncoder,
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

    # 冲突标注：同文本多标签 → 丢弃（保留会同时污染 train 与 test）
    conflict = df.groupby("text")["label"].nunique()
    bad_texts = set(conflict[conflict > 1].index)
    if bad_texts:
        before = len(df)
        df = df[~df["text"].isin(bad_texts)].reset_index(drop=True)
        print(f"[clean] 剔除标注冲突样本 {before - len(df)} 条（同文本不同标签）")

    # 语言过滤：由 --lang 独立控制，不再和增强开关耦合
    lang = getattr(args, "lang", None)
    if lang and "lang" in df.columns:
        keep = [l.strip().lower() for l in lang.split(",") if l.strip()]
        before = len(df)
        df = df[df["lang"].str.lower().isin(keep)].reset_index(drop=True)
        print(f"[clean] 语言过滤 {keep} 后: {len(df)}（原 {before}）")

    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    df.to_csv(
        os.path.join(Config.OUTPUT_PATH, "clean_data.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    print(f"[clean] 清洗后样本数: {len(df)}｜类别数: {df['label'].nunique()}")
    return df


# ────────────────────────── 划分 ──────────────────────────
def _stratify_labels(df: pd.DataFrame, args) -> Optional[pd.Series]:
    """分层标签：默认按 label；--stratify-lang 时按 label×lang。"""
    if getattr(args, "stratify_lang", False) and "lang" in df.columns:
        return df["label"].astype(str) + "|" + df["lang"].astype(str)
    return df["label"]


def stage_split(args, df: Optional[pd.DataFrame] = None, folds: int = 1) -> dict:
    if df is None:
        df = pd.read_csv(os.path.join(Config.OUTPUT_PATH, "clean_data.csv"))

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
    print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)} labels={num_labels}")

    if Config.WRITE_FILE:
        for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
            cols = ["text", "label", "label_id"] + (["lang"] if "lang" in d.columns else [])
            d[cols].to_csv(os.path.join(Config.OUTPUT_PATH, f"{name}.csv"),
                           index=False, encoding="utf-8-sig")
        with open(Config.LABELMAP_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "label2id": label2id, 
                    "id2label": id2label, 
                    "num_labels": num_labels
                },
                f, ensure_ascii=False, indent=2
            )

    return {"train": train_df, "val": val_df, "test": test_df,
            "label2id": label2id, "id2label": id2label, "num_labels": num_labels}


# ────────────────────────── 增强（仅作用于 train）──────────────────────────
def oversample_minority(
    df: pd.DataFrame,
    aug_factor: int = 2,
    minority_threshold: float = 0.8
) -> pd.DataFrame:
    """仅对少数类做 EDA 过采样。阈值改为「相对类均值的比例」，适配任意规模数据。"""
    counts = df["label"].value_counts()
    mean_n = counts.mean()
    thr = minority_threshold * mean_n
    new_rows = []
    for _, row in df.iterrows():
        new_rows.append(row)
        if counts[row["label"]] < thr:
            for _ in range(aug_factor):
                aug = row.copy()
                aug["text"] = augmentor.eda_combine(row["text"], label=row["label"])
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
    split["train"] = train_df
    print(f"[aug] 训练集过采样: {before} → {len(train_df)}（仅 train，val/test 保持原样）")

    if Config.WRITE_FILE:
        cols = ["text", "label", "label_id"] + (["lang"] if "lang" in train_df.columns else [])
        train_df[cols].to_csv(
            os.path.join(Config.OUTPUT_PATH, "train_aug.csv"),
            index=False, encoding="utf-8-sig"
        )
    return split


# ────────────────────────── 编码 ──────────────────────────
def _get_model_path() -> str:
    cache = Path(Config.MODEL_CACHE_DIR)
    model_path = cache / Config.MODEL_NAME.replace("/", "_")
    if model_path.exists() and any(model_path.iterdir()):
        return str(model_path)
    if not Config.MODEL_AUTO_DOWNLOAD:
        raise FileNotFoundError(f"模型未找到: {model_path}")
    print(f"[model] 首次运行，正在下载 {Config.MODEL_NAME} (~130MB)...")
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(Config.MODEL_NAME)
    st.save(str(model_path))
    print(f"[model] 下载完成 → {model_path}")
    return str(model_path)


def _load_onnx_encoder(onnx_path: str, tokenizer_dir: Optional[str] = None) -> ONNXEncoder:
    if tokenizer_dir:
        from transformers import AutoTokenizer
        if os.path.isdir(tokenizer_dir) and any(
                f for f in os.listdir(tokenizer_dir) if f.startswith("tokenizer")):
            return ONNXEncoder(
                onnx_path=onnx_path,
                tokenizer_dir=tokenizer_dir,
                tokenizer_name=f"sentence-transformers/{Config.MODEL_NAME}"
            )
        os.makedirs(tokenizer_dir, exist_ok=True)
        try:
            tok = AutoTokenizer.from_pretrained(f"sentence-transformers/{Config.MODEL_NAME}")
            tok.save_pretrained(tokenizer_dir)  # 关键：save 而非只设 cache_dir
            return ONNXEncoder(
                onnx_path=onnx_path,
                tokenizer_dir=tokenizer_dir,
                tokenizer_name=f"sentence-transformers/{Config.MODEL_NAME}"
            )
        except Exception as e:
            print(f"[embed] tokenizer 本地化失败（{e}），改为在线加载")
    return ONNXEncoder(
        onnx_path=onnx_path,
        tokenizer_dir=None,
        tokenizer_name=f"sentence-transformers/{Config.MODEL_NAME}"
    )


def _load_st_model(device_str: str):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(_get_model_path())
    if device_str == "cuda":
        st = st.to(torch.device("cuda"))
    return st


def stage_embed(args, split: Optional[dict] = None, backend: str = "auto") -> dict:
    if split is None:
        split = {
            k: pd.read_csv(os.path.join(Config.OUTPUT_PATH, f"{k}.csv"))
            for k in ["train", "val", "test"]
        }

    if backend == "auto":
        backend = "onnx" if os.path.exists(Config.onnx_encoder_path()) else "pytorch"

    if backend == "onnx":
        print(f"[embed] 使用 ONNX 编码器: {Config.onnx_encoder_path()}")
        encoder = _load_onnx_encoder(
            Config.onnx_encoder_path(),
            Config.onnx_encoder_tokenizer_dir()
        )

        def encode_list(texts):
            return encoder.encode(texts, batch_size=args.batch_size, normalize=True)
    else:
        device_str = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
        print(f"[embed] using device: {device_str}")
        st = _load_st_model(device_str)
        chunked = ChunkedEncoder(st, max_tokens=args.chunk_size,
                                 overlap=args.chunk_overlap, strategy=args.pool)

        def encode_list(texts):
            out = np.zeros((len(texts), Config.INPUT_DIM), dtype=np.float32)
            # 长文本判定改为按字符数阈值（与 tokenizer 解耦），短文本整批编码
            long_idx = [i for i, t in enumerate(texts) if len(t) > args.chunk_size * 4]
            short_idx = [i for i in range(len(texts)) if i not in set(long_idx)]
            if short_idx:
                v = st.encode([texts[i] for i in short_idx],
                              batch_size=args.batch_size,
                              convert_to_numpy=True,
                              show_progress_bar=False,
                              normalize_embeddings=True)  # 修复：与 ONNX 分支统一 L2 归一化
                for j, i in enumerate(short_idx):
                    out[i] = v[j].astype(np.float32)
            for i in long_idx:
                v = chunked.encode(texts[i])
                n = np.linalg.norm(v)
                out[i] = (v / n if n > 1e-9 else v).astype(np.float32)
            return out

    result: Dict[str, np.ndarray] = {}
    for name in ["train", "val", "test"]:
        texts = split[name]["text"].astype(str).tolist()
        print(f"[embed] 编码 {name} ({len(texts)} 条)...")
        result[f"X_{name}"] = encode_list(texts).astype(np.float32)
        result[f"y_{name}"] = split[name]["label_id"].values.astype(np.int64)

    np.savez_compressed(Config.EMBED_PATH, **result)  # 压缩存储，节省磁盘
    print(f"[embed] 已保存 → {Config.EMBED_PATH}")
    return result


# ────────────────────────── 训练 ──────────────────────────
def _build_loaders(emb: dict, args, num_workers: int = 0):
    loaders = {}
    for name in ["train", "val", "test"]:
        ds = TensorDataset(torch.from_numpy(np.asarray(emb[f"X_{name}"], dtype=np.float32)),
                           torch.from_numpy(np.asarray(emb[f"y_{name}"], dtype=np.int64)))
        # 数据已在内存：num_workers>0 只会带来进程间拷贝开销
        loaders[name] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(name == "train"),
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
    """EMA warmup：前几个 epoch 用更小的 decay，避免早期滞后。"""
    if warmup <= 0 or epoch >= warmup:
        return base
    return min(base, (1 + epoch) / (warmup + epoch))


def _metric(y_true, y_pred, num_labels: int, name: str) -> float:
    if name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def stage_train(args, emb: Optional[dict] = None, num_labels: Optional[int] = None):
    if emb is None:
        emb = dict(np.load(Config.EMBED_PATH))
    if num_labels is None:
        with open(Config.LABELMAP_PATH, encoding="utf-8") as f:
            num_labels = json.load(f)["num_labels"]

    set_seed(getattr(args, "seed", Config.SEED))
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    loaders = _build_loaders(emb, args, num_workers=getattr(args, "num_workers", Config.NUM_WORKERS))

    model = LightEmotionClassifier(
        input_dim=Config.INPUT_DIM,
        num_classes=num_labels,
        hidden_dim=args.hidden_dim,      # 修复：CLI 参数此前完全不生效
        dropout=args.dropout,
        stochastic_p=Config.STOCHASTIC_P,
    ).to(device)

    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    ema_model = copy.deepcopy(model)

    counts = np.bincount(np.asarray(emb["y_train"]).astype(int), minlength=num_labels)
    weights = torch.tensor(
        class_weights(counts, Config.CLASS_WEIGHT, Config.EFFECTIVE_BETA), dtype=torch.float
    ).to(device)
    print(f"[train] 类别权重策略: {Config.CLASS_WEIGHT} → {np.round(weights.cpu().numpy(), 3).tolist()}")

    criterion = FocalLoss(
        gamma=Config.FOCAL_GAMMA if Config.USE_FOCAL_LOSS else 0.0,
        weight=weights,
        label_smoothing=Config.LABEL_SMOOTHING,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.SCHED_T0, T_mult=Config.SCHED_T_MULT, eta_min=Config.SCHED_ETA_MIN
    )

    use_amp = device.type == "cuda"
    scaler = _make_scaler(device.type, use_amp)

    best_score, bad_epochs, history = -1.0, 0, []
    metric_name = getattr(args, "metric", Config.METRIC)
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            mixed_xb, ya, yb2, lam = mixup(xb, yb, alpha=Config.MIXUP_ALPHA)

            if use_amp:
                with _amp_ctx(device.type, True):
                    logits, _, _ = model(mixed_xb)
                    loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb2)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _, _ = model(mixed_xb)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
                optimizer.step()

            update_ema(model, ema_model, _ema_decay(epoch, Config.EMA_DECAY, Config.EMA_WARMUP_EPOCHS))

            tot_loss += loss.item() * len(yb)
            correct += (logits.float().argmax(1) == yb).sum().item()
            total += len(yb)

        scheduler.step(epoch - 1)  # 修复：从 0 起计数，保证 T_0 周期正确

        ema_model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in loaders["val"]:
                xb = xb.to(device, non_blocking=True)
                pred = ema_model(xb)[0].argmax(1).cpu().numpy()
                y_pred.extend(pred.tolist())
                y_true.extend(yb.numpy().tolist())
        score = _metric(y_true, y_pred, num_labels, metric_name)
        acc = float(accuracy_score(y_true, y_pred))

        history.append((epoch, tot_loss / max(1, total), correct / max(1, total), score, acc))
        print(f"[train] ep {epoch:3d} | loss {history[-1][1]:.4f} | train {history[-1][2]:.4f} "
              f"| {metric_name} {score:.4f} | acc {acc:.4f} "
              f"| lr {optimizer.param_groups[0]['lr']:.2e}")

        if score > best_score + 1e-5:
            best_score, bad_epochs = score, 0
            torch.save(
                {
                    "state_dict": ema_model.state_dict(),
                    "num_labels": num_labels,
                    "hidden_dim": args.hidden_dim,
                    "id2label": json.load(open(Config.LABELMAP_PATH, encoding="utf-8"))["id2label"]
                },
                Config.MODEL_PATH
            )
            print(f"[train] 保存最佳模型 @ ep {epoch}（{metric_name}={score:.4f}）")
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop:
                print(f"[train] 早停 @ ep {epoch} (best {metric_name}={best_score:.4f})")
                break

    print(f"[train] 最佳 {metric_name}: {best_score:.4f} → {Config.MODEL_PATH}")
    if Config.WRITE_FILE:
        with open(os.path.join(Config.OUTPUT_PATH, "train_history.json"), "w", encoding="utf-8") as f:
            json.dump({
                "metric": metric_name,
                "best": best_score,
                "history": [{
                    "epoch": h[0],
                    "loss": h[1],
                    "train_acc": h[2],
                    metric_name: h[3],
                    "val_acc": h[4]
                } for h in history]
            }, f, ensure_ascii=False, indent=2)
    return model


# ────────────────────────── 评估 ──────────────────────────
def stage_eval(args, emb: Optional[dict] = None, id2label: Optional[dict] = None):
    if emb is None:
        emb = dict(np.load(Config.EMBED_PATH))
    if id2label is None:
        with open(Config.LABELMAP_PATH, encoding="utf-8") as f:
            id2label = json.load(f)["id2label"]
    num_labels = len(id2label)

    device = torch.device("cpu")
    ckpt = torch.load(Config.MODEL_PATH, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    hidden_dim = ckpt.get("hidden_dim", Config.HIDDEN_DIM)

    model = LightEmotionClassifier(
        input_dim=Config.INPUT_DIM, num_classes=num_labels,
        hidden_dim=hidden_dim, dropout=0.0,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    X_test = torch.from_numpy(np.asarray(emb["X_test"], dtype=np.float32))
    y_test = np.asarray(emb["y_test"]).astype(int)
    with torch.no_grad():
        logits, _, _ = model(X_test)
        probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(1)

    acc = accuracy_score(y_test, preds)
    mf1 = f1_score(y_test, preds, average="macro", zero_division=0)
    print(f"\n[eval] Test Accuracy: {acc:.4f} | Macro-F1: {mf1:.4f}")
    id2label_i = {
        int(k): v 
        for k, v in id2label.items()
    }
    target_names = [id2label_i[i] for i in range(num_labels)]
    report = classification_report(y_test, preds, target_names=target_names, digits=4, zero_division=0)
    print(report)

    if Config.WRITE_FILE:
        with open(os.path.join(Config.OUTPUT_PATH, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "accuracy": float(acc), 
                    "macro_f1": float(mf1),
                    "report": report
                },
                f, ensure_ascii=False, indent=2
            )
    return {
        "accuracy": float(acc), 
        "macro_f1": float(mf1)
    }


# ────────────────────────── ONNX 导出 ──────────────────────────
def stage_onnx(args, num_labels: Optional[int] = None) -> None:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if num_labels is None:
        with open(Config.LABELMAP_PATH, encoding="utf-8") as f:
            num_labels = json.load(f)["num_labels"]

    ckpt = torch.load(Config.MODEL_PATH, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    hidden_dim = ckpt.get("hidden_dim", Config.HIDDEN_DIM)

    model = LightEmotionClassifier(
        input_dim=Config.INPUT_DIM, num_classes=num_labels,
        hidden_dim=hidden_dim, dropout=0.0,
    ).eval()
    model.load_state_dict(state)

    dummy = torch.randn(1, Config.INPUT_DIM)
    torch.onnx.export(
        model, 
        dummy,
        Config.ONNX_PATH,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,  # 17 兼容性最好；18 在部分 runtime 上不支持 LayerNorm 融合
        do_constant_folding=True,
    )
    try:
        from onnxsim import simplify
        onnx_model = onnx.load(Config.ONNX_PATH)
        onnx_model, ok = simplify(onnx_model)
        if ok:
            onnx.save(onnx_model, Config.ONNX_PATH)
    except Exception as e:
        print(f"[onnx] onnxsim 跳过（{e}）")
    onnx.checker.check_model(onnx.load(Config.ONNX_PATH))
    print(f"[onnx] FP32 模型已导出 → {Config.ONNX_PATH}")

    # FP16（体积减半，CPU 推理回退为 fp32 计算）
    try:
        from onnxconverter_common import float16
        fp16_model = float16.convert_float_to_float16(
            onnx.load(Config.ONNX_PATH),
            keep_io_types=True
        )
        onnx.save(fp16_model, Config.ONNX_FP16_PATH)
        print(f"[onnx] FP16 模型已导出 → {Config.ONNX_FP16_PATH}")
    except Exception as e:
        print(f"[onnx] FP16 导出跳过（{e}）")

    # INT8 动态量化（分类头极小，收益主要是体积）
    quant_ok = False
    try:
        quantize_dynamic(
            model_input=Config.ONNX_PATH,
            model_output=Config.ONNX_QUANT_PATH,
            weight_type=QuantType.QInt8
        )
        quant_ok = os.path.exists(Config.ONNX_QUANT_PATH)
        print(f"[onnx] INT8 量化模型已导出 → {Config.ONNX_QUANT_PATH}")
    except Exception as e:
        print(f"[onnx] 量化失败（{e}），将使用 FP32 模型进行推理")

    # 自检：shape + 数值一致性（FP32 vs 量化）
    ref = None
    sess = ort.InferenceSession(Config.ONNX_PATH, providers=["CPUExecutionProvider"])
    for b in [1, 2, 8]:
        x = np.random.randn(b, Config.INPUT_DIM).astype(np.float32)
        out = sess.run(["logits"], {"input": x})[0]
        assert out.shape == (b, num_labels), out.shape
        if b == 8:
            ref = out
    if quant_ok and ref is not None:
        qsess = ort.InferenceSession(Config.ONNX_QUANT_PATH, providers=["CPUExecutionProvider"])
        qout = qsess.run(["logits"], {"input": np.random.randn(8, Config.INPUT_DIM).astype(np.float32)})[0]
        max_err = float(np.abs(qout - ref).max())
        print(f"[onnx] 自检通过（batch 1/2/8，num_labels={num_labels}）；INT8 与 FP32 最大偏差 {max_err:.4f}")
    else:
        print(f"[onnx] 自检通过（batch 1/2/8，num_labels={num_labels}）")


# ────────────────────────── 推理 ──────────────────────────
def stage_infer(args, texts: Optional[List[str]] = None) -> None:
    import onnxruntime as ort

    if texts is None:
        texts = list(getattr(args, "infer_text", None) or [])
    if len(texts) == 0:
        return

    with open(Config.LABELMAP_PATH, encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}

    if os.path.exists(Config.onnx_encoder_path()):
        print(f"[infer] 使用 ONNX 编码器: {Config.onnx_encoder_path()}")
        encoder = _load_onnx_encoder(
            Config.onnx_encoder_path(),
            Config.onnx_encoder_tokenizer_dir()
        )
        embs = encoder.encode(texts, normalize=True)
    else:
        device_str = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
        st = _load_st_model(device_str)
        embs = st.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        ).astype(np.float32)

    onnx_path = Config.ONNX_QUANT_PATH if os.path.exists(Config.ONNX_QUANT_PATH) else Config.ONNX_PATH
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    logits = sess.run(["logits"], {"input": embs})[0]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)

    top_k = max(1, min(int(args.top_k), probs.shape[1]))
    for i, t in enumerate(texts):
        top = np.argsort(probs[i])[::-1][:top_k]
        res = ", ".join(f"{id2label.get(int(j), j)}({probs[i][j]:.2f})" for j in top)
        print(f"[infer] 「{t}」 → {res}")


def cross_validate(args, emb: dict, num_labels: int, folds: int = 5) -> None:
    """小数据集上的可靠评估：固定测试集 + 训练集 K 折，报告均值±标准差。"""
    from sklearn.model_selection import StratifiedKFold
    X = np.concatenate([emb["X_train"], emb["X_val"]])
    y = np.concatenate([emb["y_train"], emb["y_val"]]).astype(int)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=Config.RANDOM_STATE)
    scores = []
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        fold_emb = {
            "X_train": X[tr],
            "y_train": y[tr],
            "X_val": X[va],
            "y_val": y[va],
            "X_test": emb["X_test"],
            "y_test": emb["y_test"]
        }
        print(f"\n[cv] fold {k}/{folds}")
        model = stage_train(args, fold_emb, num_labels)
        res = stage_eval(args, fold_emb)
        scores.append(res["macro_f1"])
    print(f"\n[cv] macro-F1: {np.mean(scores):.4f} ± {np.std(scores):.4f}")