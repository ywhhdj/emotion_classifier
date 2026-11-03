# scripts/overfit_test.py
import sys, json, copy
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
from train.config import Config
from train.models import build_encoder_lora, load_tokenizer, TextClassificationDataset, make_collate
from train.utils import set_seed
from torch.utils.data import DataLoader

set_seed(42)
Config.USE_LANG_TOKEN = False

df = __import__("pandas").read_csv("output/train.csv")
df = df.groupby("label_id").head(6).reset_index(drop=True)   # 18 类 × 6 ≈ 108 条
print(f"overfit 样本: {len(df)}, 类别: {df['label_id'].nunique()}")

tok = load_tokenizer(Config.encoder_dir())
model = build_encoder_lora(num_labels=18, r=16, unfreeze_from=0)
model.to("cuda" if torch.cuda.is_available() else "cpu")
device = next(model.parameters()).device

ds = TextClassificationDataset(df["text"].tolist(), df["label_id"].values, tok,
                               max_len=128, use_lang_token=False)
dl = DataLoader(ds, batch_size=16, shuffle=True,
                collate_fn=make_collate(pad_id=tok.pad_token_id or 1))

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)

for ep in range(1, 31):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for b in dl:
        ids = b["input_ids"].to(device)
        mask = b["attention_mask"].to(device)
        y = b["labels"].to(device)
        opt.zero_grad()
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    acc = correct / total
    print(f"ep {ep:2d} | loss {loss_sum/total:.4f} | acc {acc:.4f}")
    if acc >= 0.99:
        print(f"[overfit] 通过：{ep} epoch 内拟合到 99%+")
        break
else:
    print("[overfit] 失败：30 epoch 内未拟合到 99%，代码存在 bug")