from setuptools import setup, find_namespace_packages

setup(
    name="emotion-classifier",
    version="0.1.0",
    packages=find_namespace_packages(include=["emotion_classifier", "emotion_classifier.*"]),
    include_package_data=True,
    package_data={
        "emotion_classifier": ["data/*"],
        "emotion_classifier.data": ["*"],
    },
    install_requires=[
        "torch>=1.12",
        "onnxruntime>=1.14",
        "numpy>=1.21",
        "httpx>=0.24",
        "tqdm>=4.62",
    ],
    extras_require={
        "onnx-only": ["onnxruntime>=1.14", "numpy>=1.21", "httpx>=0.24", "tqdm>=4.62"],
        "dev": ["pytest>=7.0", "black>=23.0", "ruff>=0.1", "build>=1.0", "twine>=4.0"],
    },
    entry_points={
        "console_scripts": [
            "emotion-classify=emotion_classifier.cli:main",
        ],
    },
)
