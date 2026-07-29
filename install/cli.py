"""
命令行接口 (CLI)
用法:
    emotion-classify "脸颊泛红，偷偷瞄了你一眼"
    emotion-classify --file input.txt --top-k 5 --backend onnx --json
    emotion-classify --batch "text1" "text2" "text3" --verbose
"""
import argparse
import json
import sys
from pathlib import Path

def build_parser():
    p = argparse.ArgumentParser(
        prog="emotion-classify",
        description="多语言情感分类 CLI（支持 19 种情感，中英日）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 输入源（三选一，但 --labels 时不需要） ──
    input_group = p.add_mutually_exclusive_group(required=False)
    input_group.add_argument("text", nargs="?", help="待分类的文本内容")
    input_group.add_argument("--file", "-f", type=str, help="从文件读取文本（每行一条）")
    input_group.add_argument("--batch", "-b", nargs="+", help="批量文本（空格分隔）")

    agrs={
        "top-k":{
            "type":int,
            "default":3,
            "help":"返回前 K 个情感标签 (默认 3)",
        },
        "backend":{
            "choices":["onnx", "pytorch", "auto"],
            "default":"auto",
            "help":"推理后端",
        },
        "model-dir":{
            "type":str,
            "default":None,
            "help":"模型目录路径",
        },
        "encoder":{
            "type":str,
            "default":None,
            "help":"SentenceTransformer 路径或名称",
        },
        "json":{
            "action":"store_true",
            "help":"以 JSON 格式输出",
        },
        "verbose":{
            "action":"store_true",
            "help":"显示详细推理信息",
        },
        "labels":{
            "action":"store_true",
            "help":"列出所有支持的标签并退出",
        },
        "interactive":{
            "action":"store_true",
            "help":"进入交互模式（逐行输入）",
        }
    }
    for k, v in agrs.items():
        if k == "top-k":
            p.add_argument(f"--{k}", "-k", **v)
        else:
            p.add_argument(f"--{k}", **v)
    return p

def load_texts(args):
    """根据参数来源收集文本列表。"""
    if args.file:
        fpath = Path(args.file)
        if not fpath.exists():
            print(f"[错误] 文件不存在: {fpath}", file=sys.stderr)
            sys.exit(1)
        texts = fpath.read_text(encoding="utf-8").strip().splitlines()
        return [t.strip() for t in texts if t.strip()]
    elif args.batch:
        return list(args.batch)
    else:
        return [args.text]

def format_output(results, texts, top_k, as_json=False, verbose=False):
    """格式化输出。"""
    if as_json:
        out = []
        for t, r in zip(texts, results):
            out.append({
                "text": t,
                "predictions": [{"label": lbl, "score": sc} for lbl, sc in r]
            })
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # 文本格式
    for i, (text, preds) in enumerate(zip(texts, results)):
        if len(texts) > 1:
            print(f"\n[{i+1}] 「{text}」")
        else:
            print(f"「{text}」")
        for rank, (lbl, sc) in enumerate(preds, 1):
            bar = "█" * int(sc * 20)
            print(f"  #{rank} {lbl:<5s} {sc:.2%}  {bar}")
        if verbose and len(preds) < top_k:
            pass  # 已足够

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    from .predictor import EmotionClassifier

    # 列出标签（仅读 label_map.json，不加载模型）
    if args.labels:
        from pathlib import Path
        import json
        if args.model_dir:
            lm_path = Path(args.model_dir) / "label_map.json"
        else:
            lm_path = Path(__file__).parent / "data" / "label_map.json"
        if not lm_path.exists():
            print(f"[错误] 找不到 {lm_path}", file=sys.stderr)
            sys.exit(1)
        with open(lm_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        labels = list(mapping["label2id"].keys())
        print(f"支持 {len(labels)} 种情感标签：")
        for i, lbl in enumerate(labels, 1):
            print(f"  {i:>2d}. {lbl}")
        return

    # 确保有输入
    if not args.text and not args.file and not args.batch:
        parser.print_help()
        print("\n[错误] 请提供文本、--file 或 --batch 参数")
        sys.exit(1)

    # 加载分类器
    try:
        clf = EmotionClassifier(
            model_dir=args.model_dir,
            backend=args.backend,
            encoder_path=args.encoder,
        )
    except FileNotFoundError as e:
        print(f"[错误] {e}", file=sys.stderr)
        print("请确认模型文件已放置于正确位置，或使用 --model-dir 指定路径。", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"[info] 后端: {clf.backend} | 标签数: {clf.num_labels}")
        print(f"[info] 支持标签: {', '.join(clf.get_labels())}\n")

    # 交互模式
    if args.interactive:
        print("交互模式（输入空行或 Ctrl+C 退出）:")
        try:
            while True:
                text = input("> ").strip()
                if not text:
                    break
                result = clf.predict(text, top_k=args.top_k)
                format_output(result, [text], args.top_k, args.json, args.verbose)
        except (KeyboardInterrupt, EOFError):
            print("\n已退出交互模式")
        return

    # 正常模式
    texts = load_texts(args)
    if not texts:
        print("[错误] 没有有效文本输入", file=sys.stderr)
        sys.exit(1)

    results = clf.predict(texts, top_k=args.top_k)
    format_output(results, texts, args.top_k, args.json, args.verbose)

if __name__ == "__main__":
    main()
