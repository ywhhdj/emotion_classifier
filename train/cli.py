import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from .config import Config  # noqa: E402
from .models import PoolingStrategy  # noqa: E402
from .train import (  # noqa: E402
    cross_validate,
    stage_augment,
    stage_clean,
    stage_embed,
    stage_eval,
    stage_infer,
    stage_onnx,
    stage_split,
    stage_train,
)
from .utils import set_seed  # noqa: E402

STAGES = ["clean", "split", "aug", "embed", "train", "cv", "eval", "onnx", "infer", "all"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="多语言情感分类 (MiniLM + 分类头 + ONNX)")
    args = {
        "stage": {"default": "all", "help": f"执行阶段，逗号分隔: {','.join(STAGES)}"},
        "data": {"default": Config.DATA_PATH, "type": str, "help": "原始 CSV 路径"},
        "output": {"default": Config.OUTPUT_PATH, "type": str, "help": "输出目录"},
        "device": {"default": Config.device(), "choices": ["cuda", "cpu"], "help": "训练设备"},
        "batch-size": {"type": int, "default": Config.BATCH_SIZE, "help": "批量大小"},
        "epochs": {"type": int, "default": Config.EPOCHS, "help": "训练轮次数"},
        "lr": {"type": float, "default": Config.LR, "help": "学习率"},
        "weight-decay": {"type": float, "default": Config.WEIGHT_DECAY, "help": "权重衰减率"},
        "hidden-dim": {"type": int, "default": Config.HIDDEN_DIM, "help": "隐藏层维度"},
        "dropout": {"type": float, "default": Config.DROPOUT, "help": "dropout 率"},
        "early-stop": {"type": int, "default": Config.EARLY_STOP, "help": "早停轮数"},
        "test-size": {"default": Config.TEST_SIZE, "type": float, "help": "测试集比例"},
        "val-size": {"default": Config.VAL_SIZE, "type": float, "help": "验证集比例"},
        "chunk-size": {"default": 120, "type": int, "help": "长文本分块 token 数"},
        "chunk-overlap": {"default": 20, "type": int, "help": "长文本分块重叠 token"},
        "pool": {"default": Config.POOL_STRATEGY, "choices": [PoolingStrategy.MEAN,
                                                              PoolingStrategy.MAX,
                                                              PoolingStrategy.WEIGHTED],
                 "help": "池化策略"},
        "num-workers": {"type": int, "default": Config.NUM_WORKERS, "help": "DataLoader 线程数"},
        "seed": {"type": int, "default": Config.SEED, "help": "随机种子"},
        "infer-text": {"nargs": "*", "default": ["脸颊泛红，偷偷瞄了你一眼",
                                                 "气鼓鼓地跺着脚瞪着你",
                                                 "i'm so excited i can hardly breathe",
                                                 "怖くて震えが止まらない"]},
        "top-k": {"type": int, "default": 3, "help": "推理 top-k"},
        "use-aug": {"type": _bool, "default": Config.USE_MULTILINGUAL_AUG,
                    "help": "是否对训练集做增强过采样"},
        "aug-factor": {"type": int, "default": Config.AUG_FACTOR, "help": "少数类增强倍率"},
        "lang": {"type": str, "default": "", "help": "只保留指定语言，逗号分隔，如 zh,en；留空=全部"},
        "stratify-lang": {"action": "store_true", "help": "划分时按 标签×语言 分层"},
        "backend": {"default": "auto", "choices": ["auto", "pytorch", "onnx"], "help": "编码器后端"},
        "metric": {"default": Config.METRIC, "choices": ["accuracy", "macro_f1"],
                   "help": "早停/选模指标（类别不平衡推荐 macro_f1）"},
        "class-weight": {"default": Config.CLASS_WEIGHT,
                         "choices": ["none", "inverse", "sqrt", "effective"],
                         "help": "类别权重策略"},
        "mixup-alpha": {"type": float, "default": Config.MIXUP_ALPHA, "help": "MixUp alpha，0 关闭"},
        "focal-gamma": {"type": float, "default": Config.FOCAL_GAMMA, "help": "Focal gamma，0 关闭"},
        "label-smoothing": {"type": float, "default": Config.LABEL_SMOOTHING, "help": "标签平滑"},
        "folds": {"type": int, "default": 0, "help": "交叉验证折数（0=不启用）"},
        "no-focal": {"action": "store_true", "help": "关闭 Focal Loss，使用普通加权 CE"},
    }
    for k, v in args.items():
        p.add_argument(f"--{k}", **v)
    return p.parse_args(argv)


def _bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")


def apply_args_to_config(args):
    Config.HIDDEN_DIM = args.hidden_dim
    Config.DROPOUT = args.dropout
    Config.WEIGHT_DECAY = args.weight_decay
    Config.MIXUP_ALPHA = args.mixup_alpha
    Config.FOCAL_GAMMA = 0.0 if args.no_focal else args.focal_gamma
    Config.USE_FOCAL_LOSS = not args.no_focal and args.focal_gamma > 0 #type: ignore
    Config.LABEL_SMOOTHING = args.label_smoothing
    Config.CLASS_WEIGHT = args.class_weight
    Config.METRIC = args.metric
    Config.AUG_FACTOR = args.aug_factor
    Config._DEVICE = args.device

    if args.data:
        Config.DATA_PATH = args.data
    if args.output:
        Config.OUTPUT_PATH = args.output
    Config.refresh_paths()


def main(argv=None):
    args = parse_args(argv)
    apply_args_to_config(args)
    Config.ensure_dirs()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stages = {s.strip() for s in args.stage.split(",")}
    run_all = "all" in stages

    def want(name: str) -> bool:
        return run_all or name in stages

    df_clean, split, emb = None, None, None

    if want("clean"):
        df_clean = stage_clean(args)
    if want("split"):
        split = stage_split(args, df_clean)
    if want("aug"):
        if split is None:
            from .train import stage_split as _sp
            split = _sp(args, df_clean)
            df_clean = df_clean if df_clean is not None else split["train"]
        split = stage_augment(args, split)
    if want("embed"):
        emb = stage_embed(args, split, backend=args.backend)

    def label_count(default=None):
        if split is not None:
            return split["num_labels"]
        if default is not None:
            return default
        with open(Config.LABELMAP_PATH, encoding="utf-8") as f: #type: ignore
            return json.load(f)["num_labels"]

    if want("train"):
        emb = emb if emb is not None else dict(np.load(Config.EMBED_PATH))#type: ignore
        stage_train(args, emb, label_count())
    if want("cv") and args.folds and args.folds > 1:
        emb = emb if emb is not None else dict(np.load(Config.EMBED_PATH))#type: ignore
        cross_validate(args, emb, label_count(), folds=args.folds)
    if want("eval"):
        idl = split["id2label"] if split else json.load(
            open(Config.LABELMAP_PATH, encoding="utf-8"))["id2label"] #type: ignore
        stage_eval(args, emb, idl)
    if want("onnx"):
        stage_onnx(args, label_count())
    if "infer" in stages or (run_all and args.infer_text):
        stage_infer(args, args.infer_text)
