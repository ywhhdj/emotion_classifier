"""兼容旧安装方式（python setup.py / pip install .）。

新项目请用 pyproject.toml（maturin 构建，含 Rust 内核）：
    pip install .            # 需要 Rust 工具链
    EMOTION_BACKEND=onnx pip install .   # 纯 Python 回退
"""

from setuptools import find_namespace_packages, setup

setup(
    name="emotion-classifier",
    version="0.2.0",
    packages=find_namespace_packages(include=["src", "src.*"]),
    include_package_data=True,
    package_data={"src": ["data/*.json"]},
    install_requires=[
        "onnxruntime>=1.14",
        "numpy>=1.21",
        "transformers>=4.30",
        "httpx>=0.24",
    ],
    extras_require={
        "pytorch": ["torch>=1.12", "transformers>=4.30", "peft>=0.10"],
        "train": [
            "torch>=1.12", "transformers>=4.30", "peft>=0.10", "accelerate>=0.25",
            "scikit-learn>=1.0", "pandas>=1.3", "python-dotenv>=0.21",
            "onnx>=1.14", "onnxsim>=0.4", "scipy>=1.10",
        ],
        "prune": ["textpruner>=1.0", "onnxconverter-common>=1.13"],
        "augment": ["sentencepiece>=0.1.99"],
        "legacy": ["sentence-transformers>=2.2"],
        "full": [
            "torch>=1.12", "transformers>=4.30", "peft>=0.10", "accelerate>=0.25",
            "scikit-learn>=1.0", "pandas>=1.3", "python-dotenv>=0.21",
            "onnx>=1.14", "onnxsim>=0.4", "onnxconverter-common>=1.13",
            "scipy>=1.10", "textpruner>=1.0", "sentencepiece>=0.1.99",
            "sentence-transformers>=2.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "emotion-classify=src.cli:main",
        ],
    },
    # 代码大量使用 PEP 604（str | Path）与内置泛型，3.10 以下会直接报错
    python_requires=">=3.10",
)
