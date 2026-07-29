import argparse
from config import Config
from models import PoolingStrategy
import os
import sys
#追加路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="多语言情感分类 (MiniLM + 分类头 + ONNX)")
    args={
        "stage":{
            "default":"all",
            "help":"执行阶段，逗号分隔: clean,split,embed,train,eval,onnx,infer,all"
        },
        "data":{
            "default":Config.DATA_PATH,
            "help":"原始 CSV 路径",
            "type":str,
        },
        "output":{
            "default":Config.OUTPUT_PATH,
            "help":"输出目录",
            "type":str,
        },
        "device":{
            "default":Config.DEVICE,
            "choices":["cuda", "cpu"],
            "help":"训练设备",
        },
        "batch-size":{
            "type":int,
            "default":Config.BATCH_SIZE,
            "help":"批量大小",
        },
        "epochs":{
            "type":int,
            "default":Config.EPOCHS,
            "help":"训练轮次数",
        },
        "lr":{
            "type":float,
            "default":Config.LR,
            "help":"学习率",
        },
        "weight-decay":{
            "type":float,
            "default":Config.WEIGHT_DECAY,
            "help":"权重衰减率",
        },
        "warmup-ratio":{
            "type":float,
            "default":Config.WARMUP_RATIO,
            "help":"预热轮数",
        },
        "hidden-dim":{
            "type":int,
            "default":Config.HIDDEN_DIM,
            "help":"隐藏层维度",
        },
        "dropout":{
            "type":float,
            "default":Config.DROPOUT,
            "help":"dropout 率",
        },
        "early-stop":{
            "type":int,
            "default":Config.EARLY_STOP,
            "help":"早停轮数",
        },
        "test-size":{
            "default":0.15,
            "type":float,
            "help":"测试集比例",
        },
        "val-size":{
            "default":0.1,
            "type":float,
            "help":"验证集集比例",
        },
        "chunk-size":{
            "default":120,
            "type":int,
            "help":"长文本分块近似 token 数",
        },
        "chunk-overlap":{
            "default":20,
            "type":int,
            "help":"长文本分块重叠 token",
        },
        "pool":{
            "default":Config.POOL_STRATEGY,
            "help":"池化策略",
            "choices":[PoolingStrategy.MEAN, PoolingStrategy.MAX, PoolingStrategy.WEIGHTED],
        },
        "num-workers":{
            "type":int,
            "default":Config.NUM_WORKERS,
            "help":"数据加载器工作线程数",
        },
        "seed":{
            "type":int,
            "default":Config.SEED,
            "help":"随机种子",
        },
        "infer-text":{
            "nargs":"*",
            "default":[
                "脸颊泛红，偷偷瞄了你一眼",
                "气鼓鼓地跺着脚瞪着你",
                "i'm so excited i can hardly breathe",
                "怖くて震えが止まらない"
            ],
            "help":"待预测文本（可多个）",
        },
        "top-k":{
            "type":int,
            "default":3,
            "help":"预测结果 top-k 数，默认前 3 个类别",
        },
        "use-aug":{
            "type":bool,
            "default":Config.USE_MULTILINGUAL_AUG,
            "help":"是否使用多语言增强，默认 True",
        },
    }
    for k, v in args.items():
        p.add_argument(f"--{k}", **v)
    return p.parse_args(argv)

def main(argv=None):
    import torch
    import numpy as np
    import json
    from train import stage_train, stage_eval, stage_onnx, stage_infer, stage_clean, stage_split, stage_embed
    args = parse_args(argv)
    Config.DATA_PATH = args.data
    Config.OUTPUT_PATH = args.output
    Config.ensure_dirs()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    stages = {s.strip() for s in args.stage.split(",")}
    run_all = "all" in stages
    df_clean, split, emb = None, None, None

    if run_all or "clean" in stages:
        df_clean = stage_clean(args)
    if run_all or "split" in stages:
        split = stage_split(args, df_clean)
    if run_all or "embed" in stages:
        emb = stage_embed(args, split)
    if run_all or "train" in stages:
        if split is None:
            with open(Config.LABELMAP_PATH) as f:
                num_labels = json.load(f)["num_labels"]
        else:
            num_labels = split["num_labels"]
        stage_train(args, emb, num_labels)
    if run_all or "eval" in stages:
        if split is None:
            with open(Config.LABELMAP_PATH) as f:
                id2label = json.load(f)["id2label"]
        else:
            id2label = split["id2label"]
        stage_eval(args, emb, id2label)
    if run_all or "onnx" in stages:
        if split is None:
            with open(Config.LABELMAP_PATH) as f:
                num_labels = json.load(f)["num_labels"]
        else:
            num_labels = split["num_labels"]
        stage_onnx(args, num_labels)
    if "infer" in stages or args.infer_text:
        stage_infer(args)

    # 若仅指定了全部但不含 infer，做一次默认推理演示
    if run_all and "infer" not in stages:
        stage_infer(args, [])


if __name__ == "__main__":
    main()