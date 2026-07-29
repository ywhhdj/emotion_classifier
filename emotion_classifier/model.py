import os
from pathlib import Path
from typing import Optional

class ONNXEncoder:
    """
    用 ONNX Runtime 加载量化后的 SentenceTransformer 编码器。
    输入: 文本列表
    输出: 384 维句向量 (numpy float32)
    """
    def __init__(self, onnx_path: str, tokenizer_dir: Optional[str] = None, tokenizer_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        # 获取模型所有输入的元信息
        self.inputs_info = {i.name: i for i in self.sess.get_inputs()}
        self.output_name = self.sess.get_outputs()[0].name
        
        # 加载 tokenizer
        if tokenizer_dir and os.path.isdir(tokenizer_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode(self, texts, batch_size=32, normalize=True):
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch,
                padding=True, truncation=True, max_length=128,
                return_tensors="np"
            )
            feeds = {}
            for input_name in self.inputs_info:
                if input_name == "input_ids":
                    feeds[input_name] = enc["input_ids"].astype(np.int64)
                elif input_name == "attention_mask":
                    feeds[input_name] = enc["attention_mask"].astype(np.int64)
                elif input_name == "token_type_ids":
                    # 如果 tokenizer 没有返回 token_type_ids，则手动创建全零数组
                    tti = enc.get("token_type_ids")
                    if tti is None:
                        tti = np.zeros_like(enc["input_ids"])
                    feeds[input_name] = tti.astype(np.int64)
                else:
                    # 其他未知输入，尝试从 enc 中获取
                    if input_name in enc:
                        feeds[input_name] = enc[input_name].astype(np.int64)
                    else:
                        raise ValueError(f"ONNX 模型要求未知输入 '{input_name}'，请检查模型导出配置。")
            
            output = self.sess.run([self.output_name], feeds)[0]
            
            # 如果输出是三维（batch, seq_len, dim），做 mean pooling
            if output.ndim == 3: #type: ignore
                mask = enc["attention_mask"][:, :, None].astype(np.float32)
                output = (output * mask).sum(axis=1) / mask.sum(axis=1)
            
            all_embeddings.append(output.astype(np.float32)) #type: ignore
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-9, None)
        return embeddings

class ModelConfig:
    GITHUB_REPO = "https://github.com/ywhhdj/emotion_classifier"
    GITHUB_TAG  = "v0.1.0"

    ONNX_FILES = ["emotion_classifier.onnx", "label_map.json"]
    PT_FILES   = ["emotion_classify.pt", "label_map.json"]

    ENCODER_ONNX = "model_quint8_avx2.onnx"
    ENCODER_TOKENIZER = "paraphrase-multilingual-MiniLM-L12-v2"

    CHECKSUMS = {
        "emotion_classify.pt": "sha256:d8e3c69a2d50f8c65100f4ca88aacbff4756410801f5da788850a7aa0d46b39d",
        "emotion_classifier.onnx": "sha256:85ff97e348eec0130b3d94113a4c69d28a2cb0ff9e91d3d888d477b85eeecaf9"
    }

    @classmethod
    def default_model_dir(cls) -> Path:
        env = os.environ.get("EMOTION_MODEL_DIR")
        if env:
            return Path(env)
        return Path.home() / ".emotion_classifier" / "models"

    @classmethod
    def download_url(cls, filename: str) -> str:
        return (
            f"{cls.GITHUB_REPO}/releases/download/"
            f"{cls.GITHUB_TAG}/{filename}"
        )

from torch.nn import Module
class EmotionClassifierNet(Module):
    """轻量多语言情感分类头。"""
    def __init__(self, input_dim=384, num_classes=19, hidden_dim=256, dropout=0.3):
        super().__init__()
        from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout
        self.net = Sequential(
            Linear(input_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, max(hidden_dim // 2, num_classes)),
            BatchNorm1d(max(hidden_dim // 2, num_classes)),
            ReLU(),
            Dropout(dropout),
            Linear(max(hidden_dim // 2, num_classes), num_classes),
        )

    def forward(self, x):
        return self.net(x)