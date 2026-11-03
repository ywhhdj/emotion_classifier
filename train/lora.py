"""
LoRA（Low-Rank Adaptation）核心实现。

设计约束（按需求）：
  · rank = 8，作用于 query / value 投影层
  · 冻结编码器底层（前 8 层，完全不插 LoRA、不求梯度）
  · 最后 4 层插 LoRA，基座权重冻结，仅适配器可训练
  · 残差 MLP 分类头结构不变（proj → residual → classifier）

关键性质：
  · A ~ Kaiming 均匀初始化、B 全零 → 初始 ΔW = 0，插入后模型输出与基座**完全一致**，
    训练从基座出发，不会破坏预训练语义。
  · merge / unmerge 可逆：merge 用于导出 ONNX（把 LoRA 折进 W），
    unmerge 用于继续训练或切换任务。
  · 适配器体积极小：4 层 × 2 模块 × (8×768 + 768×8) ≈ 98K 参数 ≈ 0.38 MB。
"""
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn


class LoRALinear(nn.Module):
    """把 LoRA 适配器包在一个已有的 nn.Linear 外面。

    y = x·W0ᵀ + b + scaling · (x·Aᵀ)·Bᵀ
    其中 A ∈ R^{r×in}，B ∈ R^{out×r}，scaling = alpha / r。
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias_trainable: bool = False,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank 必须 > 0，当前 {rank}")
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear 只能包裹 nn.Linear，当前 {type(base)}")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)
        self.merged = False

        # ── 冻结基座权重 ──
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(bias_trainable)

        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.empty(out_f, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        """标准 LoRA 初始化：A 用 Kaiming 均匀，B 置零 → 初始 ΔW = 0。"""
        # 与 nn.Linear 默认初始化保持一致（Kaiming uniform, a=√5）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def delta_weight(self) -> torch.Tensor:
        """ΔW = scaling · B @ A，形状 (out, in)。"""
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.merged:
            # (x @ Aᵀ) @ Bᵀ：先降维到 r 再升回 out，比直接算 ΔW 省显存
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) #type: ignore
            out = out + lora_out * self.scaling
        return out

    @torch.no_grad()
    def merge(self) -> None:
        """把 LoRA 折进基座权重，之后 forward 不再走适配器分支。"""
        if self.merged:
            return
        self.base.weight += self.delta_weight()
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """把 LoRA 从基座权重中减去，恢复可训练状态。"""
        if not self.merged:
            return
        self.base.weight -= self.delta_weight()
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.2f}, "
                f"merged={self.merged}"
        )


# ───────────────────────── 注入 / 管理 ─────────────────────────

# XLM-R(BERT 系) 自注意力命名是 query/key/value；部分模型用 q_proj/v_proj。
# 用后缀匹配同时兼容两种写法。
_TARGET_ALIASES = {
    "query": ("query", "q_proj", "q"),
    "key": ("key", "k_proj", "k"),
    "value": ("value", "v_proj", "v"),
    "output": ("output.dense", "out_proj", "o_proj", "dense"),
}


def _resolve_targets(names: Sequence[str]) -> Tuple[str, ...]:
    """把语义名（query/value）解析成实际可能的模块后缀集合。"""
    suffixes: List[str] = []
    for n in names:
        key = n.lower().replace("_proj", "").replace("proj", "")
        aliases = _TARGET_ALIASES.get(key)
        if aliases is None:  # 用户直接给了具体名字
            aliases = (n,)
        suffixes.extend(aliases)
    # 去重保序
    seen, out = set(), []
    for s in suffixes:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return tuple(out)


def apply_lora(
    model: nn.Module,
    target_modules: Sequence[str] = ("query", "value"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    layer_indices: Optional[Sequence[int]] = None,
    num_layers: Optional[int] = None,
    verbose: bool = True,
) -> List[str]:
    """给指定层（默认最后 4 层）的 target_modules 注入 LoRA。

    Args:
        layer_indices: 要插入 LoRA 的层号列表；None 表示「最后 lora_layers 层」。
        num_layers: 编码器总层数，用于推断末尾层；None 时自动探测。

    Returns:
        被替换的模块全名列表。
    """
    suffixes = _resolve_targets(target_modules)

    # 探测层数
    if num_layers is None:
        num_layers = _infer_num_layers(model)
    if layer_indices is None:
        # 默认：最后 4 层
        keep = 4 if num_layers and num_layers >= 4 else max(1, num_layers or 1)
        layer_indices = list(range(num_layers - keep, num_layers))

    layer_set = set(int(i) for i in layer_indices)
    replaced: List[str] = []

    for mod_name, module in list(model.named_modules()):
        if not _is_target_layer(mod_name, suffixes):
            continue
        layer_id = _extract_layer_id(mod_name)
        if layer_id is None or layer_id not in layer_set:
            continue
        parent, child = _get_parent(model, mod_name)
        if parent is None or not isinstance(child, nn.Linear):
            continue
        parent._modules[child] = LoRALinear(
            child, rank=rank, alpha=alpha, dropout=dropout
        )
        replaced.append(mod_name)

    if verbose:
        print(f"[lora] 注入 {len(replaced)} 个适配器（层 {sorted(layer_set)}，"
              f"模块 {list(target_modules)}，rank={rank}, alpha={alpha}）")
        for n in replaced[:4]:
            print(f"       · {n}")
        if len(replaced) > 4:
            print(f"       · ...（共 {len(replaced)} 个）")
    return replaced


def _infer_num_layers(model: nn.Module) -> int:
    """从 encoder.layer / encoder.block 推断层数。"""
    for attr in ("encoder", "transformer", "model"):
        enc = getattr(model, attr, None)
        if enc is None:
            continue
        for layer_attr in ("layer", "layers", "block", "blocks", "h"):
            layers = getattr(enc, layer_attr, None)
            if layers is not None:
                try:
                    return len(layers)
                except TypeError:
                    pass
    # 兜底：扫命名里出现过的最大层号
    max_id = -1
    for name, _ in model.named_modules():
        lid = _extract_layer_id(name)
        if lid is not None:
            max_id = max(max_id, lid)
    return max_id + 1 if max_id >= 0 else 0


def _extract_layer_id(module_name: str) -> Optional[int]:
    """从 '...encoder.layer.3.attention.self.query' 里取出 3。"""
    parts = module_name.split(".")
    for i, p in enumerate(parts):
        if p in ("layer", "layers", "block", "blocks", "h") and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def _is_target_layer(module_name: str, suffixes: Tuple[str, ...]) -> bool:
    tail = module_name.split(".")[-1]
    if tail in suffixes:
        return True
    # 兼容 'attention.self.query' / 'attention.output.dense'
    return module_name.endswith(suffixes)


def _get_parent(model: nn.Module, path: str):
    """返回 (父模块, 最后一节名字)。"""
    parts = path.split(".")
    if len(parts) < 2:
        return None, None
    parent = model
    for p in parts[:-1]:
        parent = parent._modules.get(p)
        if parent is None:
            return None, None
    return parent, parts[-1]


# ───────────────────────── 冻结策略 ─────────────────────────
def freeze_encoder(
    model: nn.Module,
    train_bottom_layers: int = 0,
    trainable: Sequence[str] = (),
    verbose: bool = True,
) -> Dict[str, int]:
    """冻结编码器：默认只留 LoRA 适配器可训练。

    Args:
        train_bottom_layers: 允许训练的最底部层数（0 = 全冻）。
        trainable: 额外可训练的关键字（如 'LayerNorm'、'layer_norm'）。
    """
    # 1) 先冻结所有基座参数
    for p in model.parameters():
        p.requires_grad_(False)

    # 2) 放开 LoRA 适配器
    n_lora = 0
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
            n_lora += p.numel()

    # 3) 可选：放开底层（前 N 层）
    if train_bottom_layers > 0:
        for name, p in model.named_parameters():
            lid = _extract_layer_id(name)
            if lid is not None and lid < train_bottom_layers:
                p.requires_grad_(True)

    # 4) 可选：放开归一化层
    for kw in trainable:
        for name, p in model.named_parameters():
            if kw.lower() in name.lower():
                p.requires_grad_(True)

    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        total = sum(p.numel() for p in model.parameters())
        print(f"[lora] 可训练 {trainable_total:,} / {total:,} 参数 "
              f"({trainable_total / max(total, 1) * 100:.3f}%)，其中 LoRA {n_lora:,}")
    return {
        "lora_params": n_lora,
        "trainable_params": trainable_total
    }


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """只抽出 LoRA 参数（用于保存 / 分发小适配器）。"""
    return {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def load_lora_state_dict(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
    strict: bool = True
) -> None:
    """加载 LoRA 参数（strict=False 时只加载能匹配上的）。"""
    missing = model.load_state_dict(state, strict=False)
    if strict:
        unexpected = [k for k in state if k not in model.state_dict()]
        if unexpected:
            raise KeyError(f"存在无法匹配的 LoRA 键: {unexpected[:5]}")
    return missing  # type: ignore


def set_lora_merged(model: nn.Module, merged: bool) -> None:
    """整体切换 merge 状态（导出 ONNX 前调用 merge，训练时保持 unmerged）。"""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge() if merged else m.unmerge()


def count_lora_modules(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, LoRALinear))


def lora_params_report(model: nn.Module) -> Dict[str, int]:
    """给出 LoRA / 基座 / 分类头的参数量分布，便于写进报告。"""
    lora = sum(p.numel() for n, p in model.named_parameters()
               if "lora_A" in n or "lora_B" in n)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "lora_params": lora,
        "trainable_params": trainable,
        "total_params": total,
        "lora_modules": count_lora_modules(model),
        "trainable_ratio": trainable / max(total, 1), # type: ignore
    }