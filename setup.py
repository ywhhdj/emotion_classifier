from setuptools import setup, find_namespace_packages

setup(
    name="emotion-classifier",
    version="0.1.0",
    packages=find_namespace_packages(include=["src", "src.*"]),
    include_package_data=True,
    package_data={
        "src": ["data/*"],
        "src.data": ["*"],
    },
    install_requires=[
        'onnxruntime>=1.14',
        'numpy>=1.21',
        'transformers>=4.30',
        'httpx>=0.24'
    ],
    extras_require={
        'pytorch': ['torch>=1.12', 'sentence-transformers>=2.2'],
        'full': ['torch>=1.12', 'sentence-transformers>=2.2',
                 'scikit-learn>=1.0', 'pandas>=1.3', 'imbalanced-learn>=0.10'],
    },
    entry_points={
        "console_scripts": [
            "emotion-classify=emotion_classifier.cli:main",
        ],
    },
    python_requires='>=3.8'
)
