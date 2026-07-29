from setuptools import setup, find_packages

from setuptools import setup, find_packages

setup(
    name="emotion-classifier",
    version="0.1.0",
    description="多语言情感分类模型包（19 种情感，支持中英日，ONNX/PyTorch 双后端）",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "emotion_classifier": ["data/*"],
    },
    install_requires=[
        "torch>=1.12",
        "sentence-transformers>=2.2",
        "onnxruntime>=1.14",
        "numpy>=1.21",
        "scikit-learn>=1.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "emotion-classify=emotion_classifier.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)