import sys
sys.path.insert(0, ".")
from train.models import build_encoder_lora
from train.config import Config

model = build_encoder_lora(num_labels=18, model_name="models/xlmr-zh-en-ja",
                            r=16, unfreeze_from=0)
model.print_trainable_parameters()

trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
lora_only = [(n, c) for n, c in trainable if "lora_" in n]
head_only = [(n, c) for n, c in trainable if "classifier" in n]
print(f"\n可训练总数: {sum(c for _, c in trainable):,}")
print(f"LoRA 参数:  {sum(c for _, c in lora_only):,} ({len(lora_only)} 个张量)")
print(f"Head 参数:  {sum(c for _, c in head_only):,} ({len(head_only)} 个张量)")