import argparse
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

def cli():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--mode", choices=["train", "test"], default="train")
    known, remaining = ap.parse_known_args()
    if known.mode == "train":
        from train.cli import main
    else:
        from emotion_classifier.cli import main
    main(remaining)


if __name__ == "__main__":
    cli()