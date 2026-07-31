import math
from pathlib import Path
import re
import unicodedata
from sentence_transformers import SentenceTransformer
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import resample
from config import Config
import torch
from models import ChunkedEncoder
import os
import json
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from models import EmotionClassifierNet

class TextCleaner:
    _RE_ZERO_WIDTH = re.compile(r'[\u200b-\u200d\u2028-\u202f\ufeff]')
    _RE_WHITESPACE = re.compile(r'[\u3000]+')
    _RE_CTRL       = re.compile(r'[\x00-\x1f\x7f]')
    _RE_HTML       = re.compile(r'<[^>]+>')
    _RE_URL        = re.compile(r'https?://\S+|www\.\S+')
    _RE_DECO       = re.compile(r'[【】■◆▶☞➤]+')

    @classmethod
    def clean(cls, text: str) -> str:
        if not isinstance(text, str): return ""
        text = unicodedata.normalize("NFKC", text)
        text = "".join(ch for ch in text if ch.isprintable() or "\u4e00" <= ch <= "\u9fff")
        for r in [cls._RE_CTRL, cls._RE_ZERO_WIDTH, cls._RE_WHITESPACE,
                  cls._RE_HTML, cls._RE_URL, cls._RE_DECO]:
            text = r.sub("", text).strip()
        return text

def _get_model_path() -> str:
    cache = Path(Config.MODEL_CACHE_DIR)
    # distiluse-base-multilingual-cased-v2 下载后目录名
    model_dir_name = Config.MODEL_NAME.replace("/", "_")
    model_path = cache / model_dir_name

    if model_path.exists() and any(model_path.iterdir()):
        return str(model_path)

    if not Config.MODEL_AUTO_DOWNLOAD:
        raise FileNotFoundError(
            f"模型未找到: {model_path}\n"
            f"请手动下载 {Config.MODEL_NAME} 并放置到该目录"
        )

    # 从 HuggingFace 下载（SentenceTransformer 内部处理）
    print(f"[model] 首次运行，正在下载 {Config.MODEL_NAME} (~130MB)...")
    print(f"[model] 缓存目录: {cache}")
    st = SentenceTransformer(Config.MODEL_NAME)
    # 保存到缓存目录供后续使用
    st.save(str(model_path))
    print(f"[model] 下载完成 → {model_path}")
    return str(model_path)

def _load_st_model(device_str: str) -> SentenceTransformer:
    model_path = _get_model_path()
    st = SentenceTransformer(model_path)
    if device_str == "cuda":
        st = st.to(torch.device("cuda"))
    return st

def oversample_minority(df, min_samples=20):
    """对样本数少于 min_samples 的类别进行过采样"""
    groups = df.groupby('label')
    dfs = []
    for label, group in groups:
        if len(group) < min_samples:
            group = resample(group, replace=True, n_samples=min_samples, random_state=42)
        dfs.append(group)
    return pd.concat(dfs).reset_index(drop=True)


def stage_clean(args, df_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    if df_raw is None:
        for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
            try:
                df_raw = pd.read_csv(Config.DATA_PATH, encoding=enc)
                break
            except Exception:
                continue
        if df_raw is None:
            raise FileNotFoundError(Config.DATA_PATH)
    if "text" not in df_raw.columns or "label" not in df_raw.columns:
        raise KeyError("CSV 必须包含 text 和 label 两列")

    df = df_raw.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).apply(TextCleaner.clean)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    print(f"[clean] 清洗后样本数: {len(df)}")

    # 多语言增强
    if not getattr(args, 'use_aug', True):
        #不用则排除lang列除zh的样本
        if 'lang' in df.columns:
            df = df[df['lang'] == 'zh']
            print(f"[clean] 排除非中文后: {len(df)}")

    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    df.to_csv(os.path.join(Config.OUTPUT_PATH, "clean_data.csv"), index=False, encoding="utf-8-sig")
    df = oversample_minority(df)
    return df


def stage_split(args, df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_csv(os.path.join(Config.OUTPUT_PATH, "clean_data.csv"))
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    num_labels = len(le.classes_)
    label2id = {c: int(i) for i, c in enumerate(le.classes_)}
    id2label = {int(i): c for i, c in enumerate(le.classes_)}

    # 分层抽样，处理小类
    stratify = df["label_id"] if df["label_id"].value_counts().min() >= 2 else None
    train_df, temp_df = train_test_split(
        df, test_size=args.test_size + args.val_size,
        random_state=Config.RANDOM_STATE, stratify=stratify
    )
    val_frac = args.val_size / (args.test_size + args.val_size)
    stratify2 = temp_df["label_id"] if temp_df["label_id"].value_counts().min() >= 2 else None
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_frac,
        random_state=Config.RANDOM_STATE, stratify=stratify2
    )
    print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)}  labels={num_labels}")

    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        d[["text", "label", "label_id"]].to_csv(
            os.path.join(Config.OUTPUT_PATH, f"{name}.csv"), index=False, encoding="utf-8-sig"
        )
    if Config.WRITE_FILE:
        with open(Config.LABELMAP_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "label2id": label2id, 
                    "id2label": id2label, 
                    "num_labels": num_labels
                },
                f, 
                ensure_ascii=False,
                indent=2
            )
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "label2id": label2id, 
        "id2label": id2label, 
        "num_labels": num_labels
    }


def stage_embed(args, split: dict | None = None):
    """对 train/val/test 编码并缓存为 npz。"""
    if split is None:
        split = {
            k: pd.read_csv(os.path.join(Config.OUTPUT_PATH, f"{k}.csv"))
                 for k in ["train", "val", "test"]
        }
    device_str = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[embed] using device: {device_str}")
    st = _load_st_model(device_str)
    encoder = ChunkedEncoder(
        st,
        max_tokens=args.chunk_size,
        overlap=args.chunk_overlap,
        strategy=args.pool
    )
    def encode_list(texts):
        out = np.zeros((len(texts), Config.INPUT_DIM), dtype=np.float32)
        batch = []
        idx = []
        for i, t in enumerate(texts):
            if len(t) <= args.chunk_size * 2:   # 短文本直接走 batch
                batch.append(t)
                idx.append(i)
                if len(batch) >= args.batch_size:
                    v = st.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                    for j, vj in zip(idx, v): out[j] = vj.astype(np.float32)
                    batch, idx = [], []
            else:
                out[i] = encoder.encode(t)
        if batch:
            v = st.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            for j, vj in zip(idx, v): out[j] = vj.astype(np.float32)
        return out

    result = {}
    for name in ["train", "val", "test"]:
        texts = split[name]["text"].tolist()
        print(f"[embed] 编码 {name} ({len(texts)} 条)...")
        result[f"X_{name}"] = encode_list(texts)
        result[f"y_{name}"] = split[name]["label_id"].values.astype(np.int64)

    np.savez(Config.EMBED_PATH, **result)
    print(f"[embed] 已保存 → {Config.EMBED_PATH}")
    return result


def _build_loaders(emb: dict, args, num_labels: int):
    loaders = {}
    for name in ["train", "val", "test"]:
        ds = TensorDataset(
            torch.from_numpy(emb[f"X_{name}"]),
            torch.from_numpy(emb[f"y_{name}"])
        )
        shuffle = (name == "train")
        loaders[name] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers, 
            pin_memory=args.device=="cuda"
        )
    return loaders


def stage_train(args, emb: dict | None = None, num_labels: int | None = None):
    if emb is None: emb = np.load(Config.EMBED_PATH)
    if num_labels is None:
        with open(Config.LABELMAP_PATH) as f: num_labels = json.load(f)["num_labels"]

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    loaders = _build_loaders(emb, args, num_labels) # type: ignore

    model = EmotionClassifierNet(
        input_dim=Config.INPUT_DIM, 
        num_classes=num_labels, # type: ignore
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    # 权重初始化
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)
    if Config.USE_FOCAL_LOSS:
        from models import FocalLoss
        from collections import Counter
        label_ids = emb["y_train"]   # type: ignore
        counts = Counter(label_ids)
        weights = [1.0 / counts[i] for i in range(num_labels)] # type: ignore
        weights = torch.tensor(weights, dtype=torch.float).to(device)
        weights = weights / weights.sum() * num_labels   # type: ignore 归一化 
        criterion = FocalLoss(gamma=Config.FOCAL_GAMMA, alpha=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(loaders["train"]))
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_val, bad_epochs, history = 0.0, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()
            tot_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for xb, yb in loaders["val"]:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                pred = model(xb).argmax(1)
                v_correct += (pred == yb).sum().item()
                v_total += len(yb)
        val_acc = v_correct / max(1, v_total)
        row = (ep, tot_loss/max(1,total), correct/max(1,total), val_acc)
        history.append(row)
        print(f"[train] ep {ep:3d} | loss {row[1]:.4f} | train {row[2]:.4f} | val {row[3]:.4f}")
        scheduler.step(val_acc)
        if val_acc > best_val + 1e-5:
            best_val, bad_epochs = val_acc, 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop:
                print(f"[train] 早停 @ ep {ep} (best val={best_val:.4f})")
                break
    print(f"[train] 最佳验证准确率: {best_val:.4f} → {Config.MODEL_PATH}")
    return model


def stage_eval(args, emb: dict | None = None, id2label: dict | None = None):
    if emb is None: emb = np.load(Config.EMBED_PATH)
    if id2label is None:
        with open(Config.LABELMAP_PATH) as f: id2label = json.load(f)["id2label"]
    num_labels = len(id2label) # type: ignore

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = EmotionClassifierNet(
        input_dim=Config.INPUT_DIM,
        num_classes=num_labels,
        hidden_dim=args.hidden_dim, 
        dropout=args.dropout
    ).to(device)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    X_test = torch.from_numpy(emb["X_test"]).to(device) # type: ignore
    y_test = emb["y_test"] # type: ignore
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(1).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    print(f"\n[eval] Test Accuracy: {acc:.4f}")
    target_names = [id2label[str(i)] if str(i) in id2label else id2label[i] for i in range(num_labels)] # type: ignore
    # 统一 key 类型
    id2label_i = {int(k): v for k, v in id2label.items()} # type: ignore
    target_names = [id2label_i[i] for i in range(num_labels)] # type: ignore
    print(classification_report(y_test, preds, target_names=target_names, digits=4))
    return acc


def stage_onnx(args, num_labels: int | None = None):
    """导出 ONNX 并做推理自检。"""
    import onnx, onnxruntime as ort
    if num_labels is None:
        with open(Config.LABELMAP_PATH) as f: num_labels = json.load(f)["num_labels"]

    device = torch.device("cpu")  # ONNX 导出必须在 CPU
    model = EmotionClassifierNet(
        input_dim=Config.INPUT_DIM,
        num_classes=num_labels, # type: ignore
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    dummy = torch.randn(1, Config.INPUT_DIM)
    torch.onnx.export(
        model, 
        dummy,  # type: ignore
        Config.ONNX_PATH, 
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
        keep_initializers_as_inputs=False
    )
    onnx_model = onnx.load(Config.ONNX_PATH)
    onnx.save(onnx_model, Config.ONNX_PATH)
    onnx.checker.check_model(onnx_model)

    # 推理自检
    sess = ort.InferenceSession(Config.ONNX_PATH, providers=["CPUExecutionProvider"])
    for b in [1, 2, 8]:
        x = np.random.randn(b, Config.INPUT_DIM).astype(np.float32)
        out = sess.run(["logits"], {"input": x})[0]
        assert out.shape == (b, num_labels), out.shape # type: ignore

    print(f"[onnx] 已导出 → {Config.ONNX_PATH} (单一文件，无 .data)")
    print(f"[onnx] 自检通过 (batch 1/2/8, num_labels={num_labels})")


def stage_infer(args, texts: list[str] | None = None):
    """加载 ONNX + label_map，对输入文本做预测。"""
    import onnxruntime as ort
    if texts is None:
        texts = args.infer_text or [
            "脸颊泛红，偷偷瞄了你一眼",
            "气鼓鼓地戳着脚瞪着你",
            "i'm so excited i can hardly breathe",
            "怖くて震えが止まらない"
        ]
    if len(texts) == 0:
        return
    with open(Config.LABELMAP_PATH,encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}

    device_str = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    st = _load_st_model(device_str)
    sess = ort.InferenceSession(Config.ONNX_PATH, providers=["CPUExecutionProvider"])

    embs = st.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    logits = sess.run(["logits"], {"input": embs})[0]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True) # type: ignore
    top_k = args.top_k
    for i, t in enumerate(texts):
        top = np.argsort(probs[i])[::-1][:top_k]
        res = ", ".join(f"{id2label[int(j)]}({probs[i][j]:.2f})" for j in top)
        print(f"[infer] 「{t}」 → {res}")

