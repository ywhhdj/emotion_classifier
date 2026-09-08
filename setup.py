from setuptools import find_namespace_packages, setup

setup(
    name="emotion-classifier",
    version="0.1.1",
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
        "pytorch": ["torch>=1.12", "sentence-transformers>=2.2"],
        "full": [
            "torch>=1.12",
            "sentence-transformers>=2.2",
            "scikit-learn>=1.0",
            "pandas>=1.3",
            "python-dotenv>=0.21",
            "onnx>=1.14",
            "onnxsim>=0.4",
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

#运行打包命令
# python setup.py sdist bdist_wheel