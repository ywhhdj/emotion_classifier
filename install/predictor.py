import json
import numpy as np
from pathlib import Path

class EmotionClassifier:
    def __init__(self, model_dir=None, backend="auto", encoder_path=None):
        """
        Parameters
        ----------
        model_dir : str | Path | None
            模型文件目录，默认使用包内 data/ 目录
        backend : str
            'onnx' | 'pytorch' | 'auto'
        encoder_path : str | None
            SentenceTransformer 模型路径或名称，默认自动下载
        """
        if model_dir is None:
            model_dir = Path(__file__).parent / "data"
        self.model_dir = Path(model_dir)

        # ── 加载标签映射 ──
        label_map_path = self.model_dir / "label_map.json"
        if not label_map_path.exists():
            raise FileNotFoundError(f"找不到 label_map.json: {label_map_path}")
        with open(label_map_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
        self.label2id = mapping["label2id"]
        self.num_labels = mapping["num_labels"]

        # ── 选择后端 ──
        onnx_path = self.model_dir / "emotion_classifier.onnx"
        pt_path = self.model_dir / "best_model.pt"
        if backend == "auto":
            if onnx_path.exists():
                backend = "onnx"
            elif pt_path.exists():
                backend = "pytorch"
            else:
                raise FileNotFoundError(
                    f"未找到模型文件，请检查 {self.model_dir}"
                )
        self.backend = backend

        if backend == "onnx":
            self._init_onnx(str(onnx_path))
        else:
            self._init_pytorch(str(pt_path))

        # ── 加载句子编码器 ──
        from sentence_transformers import SentenceTransformer
        model_name = encoder_path or "paraphrase-multilingual-MiniLM-L12-v2"
        self.encoder = SentenceTransformer(model_name)

    def _init_onnx(self, onnx_path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )

    def _init_pytorch(self, pt_path: str):
        import torch
        from .model import EmotionClassifierNet
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # hidden_dim 需与训练时一致
        self.model = EmotionClassifierNet(
            input_dim=384,
            num_classes=self.num_labels,
            hidden_dim=128,
            dropout=0.0,  # 推理时关闭 dropout
        ).to(self.device)
        state = torch.load(pt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def encode(self, texts:list[str]):
        if isinstance(texts, str):
            texts = [texts]
        embs = self.encoder.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embs.astype(np.float32)

    def predict(self, texts: str | list[str], top_k:int=3):
        """
        情感分类预测。

        Parameters
        ----------
        texts : str | list[str]
        top_k : int

        Returns
        -------
        list[list[tuple[str, float]]]
            每条文本的前 top_k 个 (标签, 概率)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.encode(texts)

        if self.backend == "onnx":
            logits = self.sess.run(["logits"], {"input": embeddings})[0]
        else:
            import torch
            with torch.no_grad():
                logits = self.model(
                    torch.from_numpy(embeddings).to(self.device)
                ).cpu().numpy()

        # softmax
        logits = logits - logits.max(axis=1, keepdims=True) #type: ignore
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        results = []
        for prob_vec in probs:
            top_idx = np.argsort(prob_vec)[::-1][:top_k]
            item = [
                (self.id2label[int(i)], round(float(prob_vec[i]), 4))
                for i in top_idx
            ]
            results.append(item)
        return results

    def predict_label(self, text, top_k=1):
        return self.predict(text, top_k=top_k)[0]

    def get_labels(self):
        return list(self.label2id.keys())

    def __repr__(self):
        return (
            f"EmotionClassifier(backend='{self.backend}', "
            f"labels={self.num_labels}, model_dir='{self.model_dir}')"
        )