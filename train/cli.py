import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from .config import Config  # noqa: E402
from .models import PoolingStrategy  # noqa: E402
from .train import (  # noqa: E402
    cross_validate,
    stage_augment,
    stage_calibrate,
    stage_clean,
    stage_embed,
    stage_eval,
    stage_infer,
    stage_onnx,
    stage_split,
    stage_train,
    stage_train_e2e,
)
from .utils import set_seed  # noqa: E402

STAGES = [
    "clean", 
    "split",
    "aug",
    "embed",
    "train",
    "cv",
    "eval",
    "calib",
    "onnx",
    "infer",
    "all"
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="多语言情感分类 (XLM-R + LoRA/分类头 + ONNX)")
    args = {
        "stage": {"default": "all", "help": f"执行阶段，逗号分隔: {','.join(STAGES)}"},
        "data": {"default": Config.DATA_PATH, "type": str, "help": "原始 CSV 路径"},
        "output": {"default": Config.OUTPUT_PATH, "type": str, "help": "输出目录"},
        "device": {"default": Config.device(), "choices": ["cuda", "cpu"], "help": "训练设备"},
        "train-mode": {"default": Config.TRAIN_MODE, "choices": ["e2e", "cached"],
                       "help": "e2e=端到端(LoRA)；cached=冻结编码器预编码后只训分类头"},
        "model-name": {"default": Config.MODEL_NAME, "type": str,
                       "help": "编码器基座：本地目录或 HF hub 名（默认自动解析）"},
        "base-model-dir": {"default": Config.BASE_MODEL_DIR, "type": str,
                           "help": "训练基座目录（HF 格式，三语裁剪产物；不是 ONNX）"},
        "deploy-dir": {"default": Config.DEPLOY_DIR if hasattr(Config, "DEPLOY_DIR")
                       else "models/deploy", "type": str,
                       "help": "部署产物输出目录（src 与 Rust 只认它）"},
        "batch-size": {"type": int, "default": Config.BATCH_SIZE, "help": "批量大小"},
        "epochs": {"type": int, "default": Config.EPOCHS, "help": "训练轮次数（e2e）"},
        "lr": {"type": float, "default": Config.LR, "help": "学习率（e2e=LoRA lr）"},
        "weight-decay": {"type": float, "default": Config.WEIGHT_DECAY, "help": "权重衰减率"},
        "hidden-dim": {"type": int, "default": Config.HIDDEN_DIM, "help": "分类头隐藏层维度"},
        "dropout": {"type": float, "default": Config.DROPOUT, "help": "dropout 率"},
        "early-stop": {"type": int, "default": Config.EARLY_STOP, "help": "早停轮数"},
        "test-size": {"default": Config.TEST_SIZE, "type": float, "help": "测试集比例"},
        "val-size": {"default": Config.VAL_SIZE, "type": float, "help": "验证集比例"},
        "max-seq-len": {"type": int, "default": Config.MAX_SEQ_LEN,
                        "help": "最大序列长度（数据 max=45 字符，48 足够）"},
        "pool": {"default": Config.POOL_STRATEGY,
                 "choices": [PoolingStrategy.CLS, PoolingStrategy.MEAN,
                             PoolingStrategy.WEIGHTED, PoolingStrategy.CLS_MEAN],
                 "help": "池化策略（cached 模式）"},
        # ── LoRA ──
        "lora-r": {"type": int, "default": Config.LORA_R, "help": "LoRA rank"},
        "lora-unfreeze-from": {"type": int, "default": Config.LORA_UNFREEZE_FROM,
                               "help": "从第几层开始挂 LoRA（前层冻结）"},
        "use-lang-token": {"type": _bool, "default": Config.USE_LANG_TOKEN,
                           "help": "输入前加 <zh>/<en>/<ja>"},
        # ── prompt 完形（路径 B）──
        "use-prompt": {"type": _bool, "default": Config.USE_PROMPT,
                       "help": "冻结编码器 + prompt 完形（仅 cached 模式）"},
        # ── 回译 / 跨语言扩充 ──
        "use-bt": {"type": _bool, "default": Config.USE_BACKTRANS, "help": "回译增强"},
        "bt-factor": {"type": int, "default": Config.BT_FACTOR, "help": "每条产出的回译样本数"},
        "bt-model": {"default": Config.BT_MODEL, "type": str, "help": "翻译模型（NLLB）"},
        "cross-lingual": {"type": _bool, "default": Config.USE_CROSSLING,
                          "help": "跨语言扩充（zh/en -> ja/en）"},
        # ── 其他 ──
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
                   "help": "早停/选模指标"},
        "class-weight": {"default": Config.CLASS_WEIGHT,
                         "choices": ["none", "inverse", "sqrt", "effective"],
                         "help": "类别权重策略"},
        "mixup-alpha": {"type": float, "default": Config.MIXUP_ALPHA, "help": "MixUp alpha，0 关闭"},
        "focal-gamma": {"type": float, "default": Config.FOCAL_GAMMA, "help": "Focal gamma，0 关闭"},
        "label-smoothing": {"type": float, "default": Config.LABEL_SMOOTHING, "help": "标签平滑"},
        "folds": {"type": int, "default": 0, "help": "交叉验证折数（0=不启用）"},
        "no-focal": {"action": "store_true", "help": "关闭 Focal Loss"},
        "calibrate": {"type": _bool, "default": Config.CALIBRATE,
                      "help": "导出前做 temperature 校准"},
        "chunk-size": {"default": 120, "type": int, "help": "[DEPRECATED] 长文本分块 token 数"},
        "chunk-overlap": {"default": 20, "type": int, "help": "[DEPRECATED] 长文本分块重叠 token"},
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
    Config.USE_FOCAL_LOSS = not args.no_focal and args.focal_gamma > 0  # type: ignore
    Config.LABEL_SMOOTHING = args.label_smoothing
    Config.CLASS_WEIGHT = args.class_weight
    Config.METRIC = args.metric
    Config.AUG_FACTOR = args.aug_factor
    Config._DEVICE = args.device

    if getattr(args, "base_model_dir", None):
        Config.BASE_MODEL_DIR = args.base_model_dir
    if getattr(args, "deploy_dir", None):
        Config.DEPLOY_DIR = args.deploy_dir
    Config.MODEL_NAME = args.model_name
    Config.TRAIN_MODE = args.train_mode
    Config.MAX_SEQ_LEN = args.max_seq_len
    Config.POOL_STRATEGY = args.pool
    Config.USE_LANG_TOKEN = args.use_lang_token
    Config.USE_PROMPT = args.use_prompt
    Config.LORA_R = args.lora_r
    Config.LORA_UNFREEZE_FROM = args.lora_unfreeze_from
    Config.USE_BACKTRANS = args.use_bt
    Config.BT_FACTOR = args.bt_factor
    Config.BT_MODEL = args.bt_model
    Config.USE_CROSSLING = args.cross_lingual
    Config.CALIBRATE = args.calibrate

    # epochs / early_stop / batch_size 按模式给默认值
    if args.train_mode == "cached":
        if args.epochs == Config.EPOCHS:
            args.epochs = Config.CACHED_EPOCHS
        if args.early_stop == Config.EARLY_STOP:
            args.early_stop = Config.CACHED_EARLY_STOP
        if args.lr == Config.LORA_LR:
            args.lr = Config.CACHED_LR
        if args.batch_size == Config.BATCH_SIZE:
            args.batch_size = 64

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
            split = stage_split(args, df_clean)
            df_clean = df_clean if df_clean is not None else split["train"]
        # 跨语言扩充必须在 split 之前才不泄漏 —— 这里放在 split 之后是错的，
        # 因此 CLI 只在 --stage 显式包含 clean 且提供了 --cross-lingual 时才整体重排。
        if Config.USE_CROSSLING:
            from .crosslingual import expand_crosslingual
            src = df_clean if df_clean is not None else pd.concat(
                [split["train"], split["val"], split["test"]], ignore_index=True)
            expanded = expand_crosslingual(src, device=args.device,
                                           model_name=Config.BT_MODEL)
            df_clean = expanded
            split = stage_split(args, expanded)
            print("[cl] 注意：跨语言扩充会重排划分，group_id 相同的样本整组划分")
        split = stage_augment(args, split)
    if want("embed"):
        if args.train_mode == "e2e":
            print("[embed] e2e 模式跳过预编码缓存（编码器参与训练，缓存必然失效）")
        else:
            emb = stage_embed(args, split, backend=args.backend)

    def label_count(default=None):
        if split is not None:
            return split["num_labels"]
        if default is not None:
            return default
        with open(Config.LABELMAP_PATH, encoding="utf-8") as f:  # type: ignore
            return json.load(f)["num_labels"]

    if want("train"):
        enc = Config.encoder_dir()
        print(f"[base] 训练基座: {enc}"
              f"（{'三语裁剪产物' if Config.is_trimmed() else '原始 XLM-R，未裁剪'}）")
        if args.train_mode == "e2e":
            if split is None:
                raise SystemExit("e2e 模式需要 split 阶段（--stage 至少包含 clean,split,train）")
            stage_train_e2e(args, split, label_count())
        else:
            emb = emb if emb is not None else dict(np.load(Config.EMBED_PATH))  # type: ignore
            stage_train(args, emb, label_count())
    if want("cv") and args.folds and args.folds > 1:
        if args.train_mode == "e2e":
            cross_validate(args, {}, label_count(), folds=args.folds, split=split)
        else:
            emb = emb if emb is not None else dict(np.load(Config.EMBED_PATH))  # type: ignore
            cross_validate(args, emb, label_count(), folds=args.folds)
    if want("eval"):
        idl = split["id2label"] if split else json.load(
            open(Config.LABELMAP_PATH, encoding="utf-8"))["id2label"]  # type: ignore
        stage_eval(args, emb, idl, split)
    if want("calib") and Config.CALIBRATE:
        stage_calibrate(args, split)
    if want("onnx"):
        stage_onnx(args, label_count())
    if "infer" in stages or (run_all and args.infer_text):
        stage_infer(args, args.infer_text)


if __name__ == "__main__":
    main()
