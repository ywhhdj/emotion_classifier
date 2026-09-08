import os
from typing import Literal
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

mode:Literal["train","test"]="test"
if mode=="train":
    from train.cli import main
else:
    from src.cli import main

if __name__ == "__main__":
    main()