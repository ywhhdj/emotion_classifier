"""
命令行接口 (CLI)
用法:
    emotion-classify "脸颊泛红，偷偷瞄了你一眼"
    emotion-classify --file input.txt --top-k 5 --backend onnx --json
    emotion-classify --batch "text1" "text2" "text3" --verbose
    emotion-classify --update               # 强制更新模型
    emotion-classify --info                 # 显示模型状态
"""
import argparse
import json
import os
import sys
from pathlib import Path

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS # type: ignore
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.normpath(os.path.join(base_path, relative_path))

def build_parser():
    p = argparse.ArgumentParser(
        prog="emotion-classify",
        description="多语言情感分类 CLI（支持 19 种情感，中英日，自动下载模型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 输入源 ──
    input_group = p.add_mutually_exclusive_group(required=False)
    input_group.add_argument("text", nargs="?", help="待分类的文本内容")
    input_group.add_argument("--file", "-f", type=str, help="从文件读取文本（每行一条）")
    input_group.add_argument("--batch", "-b", nargs="+", help="批量文本（空格分隔）")

    # ── 可选参数 ──
    p.add_argument("--top-k", "-k", type=int, default=3, help="返回前 K 个情感标签 (默认 3)")
    p.add_argument("--backend", choices=["onnx", "pytorch", "auto"], default="auto", help="推理后端")
    p.add_argument("--model-dir", type=str, default=get_resource_path("models"), help="模型目录路径（默认 ~/.emotion_classifier/models）")
    p.add_argument("--encoder", type=str, default=None, help="SentenceTransformer 路径或名称")
    p.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    p.add_argument("--verbose", "-v", action="store_true", help="显示详细推理信息")
    p.add_argument("--labels", action="store_true", help="列出所有支持的标签并退出")
    p.add_argument("--interactive", "-i", action="store_true", help="进入交互模式（逐行输入）")
    p.add_argument("--update", action="store_true", help="强制重新下载模型文件")
    p.add_argument("--info", action="store_true", help="显示模型状态信息")
    p.add_argument("--no-download", action="store_true", help="禁止自动下载（离线模式）")
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

    for i, (text, preds) in enumerate(zip(texts, results)):
        if len(texts) > 1:
            print(f"\n[{i+1}] 「{text}」")
        else:
            print(f"「{text}」")
        for rank, (lbl, sc) in enumerate(preds, 1):
            bar = "█" * int(sc * 20)
            print(f"  #{rank} {lbl:<5s} {sc:.2%}  {bar}")

def _load_label_map(model_dir=None):
    if model_dir:
        lm_path = Path(model_dir) / "label_map.json"
    else:
        lm_path = Path.home() / ".emotion_classifier" / "models" / "label_map.json"
    if not lm_path.exists():
        lm_path = Path(get_resource_path("data/label_map.json"))
    return lm_path



def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    from .predictor import EmotionClassifier

    if args.labels:
        lm_path = _load_label_map(args.model_dir)
        if not lm_path.exists():
            print(f"[错误] 找不到 {lm_path}", file=sys.stderr)
            print("请先运行一次 emotion-classify 以下载模型文件", file=sys.stderr)
            sys.exit(1)
        with open(lm_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        labels = list(mapping["label2id"].keys())
        print(f"支持 {len(labels)} 种情感标签：")
        for i, lbl in enumerate(labels, 1):
            print(f"  {i:>2d}. {lbl}")
        return

    # ── --info: 显示模型状态 ──
    if args.info:
        clf = EmotionClassifier(
            model_dir=args.model_dir,
            backend=args.backend,
            auto_download=not args.no_download,
        )
        info = clf.model_info()
        print("=== Emotion Classifier 模型状态 ===")
        print(f"  后端: {info['backend']}")
        print(f"  目录: {info['model_dir']}")
        print(f"  标签数: {info['num_labels']}")
        for fname, finfo in info["files"].items():
            status = f"✅ {finfo['size_mb']}MB" if finfo["exists"] else "❌ 缺失"
            print(f"  {fname:30s} {status}")
        return

    # ── --update: 强制更新 ──
    if args.update:
        clf = EmotionClassifier(
            model_dir=args.model_dir,
            backend=args.backend,
            auto_download=True,
        )
        clf.update_models()
        print("模型更新完成")
        return

    # 确保有输入
    if not args.text and not args.file and not args.batch:
        if not (args.info or args.labels or args.update):
            parser.print_help()
            print("\n[提示] 使用 emotion-classify \"文本\" 开始预测")
            sys.exit(0)

    # ── 正常预测流程 ──
    try:
        clf = EmotionClassifier(
            model_dir=args.model_dir,
            backend=args.backend,
            encoder_path=args.encoder,
            auto_download=not args.no_download,
        )
    except FileNotFoundError as e:
        print(f"[错误] {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        info = clf.model_info()
        print(f"[info] 后端: {clf.backend} | 标签数: {clf.num_labels}")
        print(f"[info] 模型目录: {info['model_dir']}")
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

    texts = load_texts(args)
    if not texts:
        print("[错误] 没有有效文本输入", file=sys.stderr)
        sys.exit(1)

    results = clf.predict(texts, top_k=args.top_k)
    format_output(results, texts, args.top_k, args.json, args.verbose)