from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np
import os

class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        pass

class ONNXEncoder(BaseEncoder):
    EXPECTED_DIM = 384

    def __init__(
        self,
        onnx_path: str,
        tokenizer_dir: Optional[str] = None,
        tokenizer_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_tokens: int = 120,
        overlap: int = 20,
    ):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.max_tokens = max_tokens
        self.overlap = overlap
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inputs_info = {i.name: i for i in self.sess.get_inputs()}
        self.output_name = self.sess.get_outputs()[0].name

        # 加载 tokenizer
        if tokenizer_dir and os.path.isdir(tokenizer_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # 验证输出维度
        dummy = self.tokenizer(["test"], padding=True, truncation=True, max_length=16, return_tensors="np")
        feeds = {}
        for input_name in self.inputs_info:
            if input_name == "input_ids":
                feeds[input_name] = dummy["input_ids"].astype(np.int64)
            elif input_name == "attention_mask":
                feeds[input_name] = dummy["attention_mask"].astype(np.int64)
            elif input_name == "token_type_ids":
                tti = dummy.get("token_type_ids")
                if tti is None:
                    tti = np.zeros_like(dummy["input_ids"])
                feeds[input_name] = tti.astype(np.int64)
            else:
                # 对于其他输入（如 position_ids），尝试从 dummy 获取，否则填充零
                if input_name in dummy:
                    feeds[input_name] = dummy[input_name].astype(np.int64)
                else:
                    feeds[input_name] = np.zeros_like(dummy["input_ids"], dtype=np.int64)
        out = self.sess.run([self.output_name], feeds)[0]
        if out.ndim == 3: #type: ignore
            out = out.mean(axis=1) #type: ignore
        actual_dim = out.shape[-1] #type: ignore
        if actual_dim != self.EXPECTED_DIM:
            raise ValueError(
                f"编码器输出维度应为 {self.EXPECTED_DIM}，实际为 {actual_dim}。"
                f"请提供输出维度为 {self.EXPECTED_DIM} 的 ONNX 编码器。"
            )

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embs = []
            for text in batch:
                emb = self.__encode_single(text, normalize=False)
                batch_embs.append(emb)
            batch_embs = np.stack(batch_embs, axis=0)
            all_embeddings.append(batch_embs)

        embeddings = np.concatenate(all_embeddings, axis=0)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-9, None)
        return embeddings.astype(np.float32)

    def __encode_single(self, text: str, normalize: bool) -> np.ndarray:
        if self.sess is None:
            raise ValueError("ONNX 编码器未初始化，无法编码单个文本。")
        tokens = self.tokenizer.tokenize(text)
        step = self.max_tokens - self.overlap
        if len(tokens) <= self.max_tokens:
            chunks = [text]
        else:
            chunks = []
            for start in range(0, len(tokens), step):
                chunk_tokens = tokens[start:start + self.max_tokens]
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks.append(chunk_text)

        # 编码每个分块
        chunk_vecs = []
        for chunk in chunks:
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="np"
            )
            feeds = {}
            for input_name in self.inputs_info:
                if input_name == "input_ids":
                    feeds[input_name] = enc["input_ids"].astype(np.int64)
                elif input_name == "attention_mask":
                    feeds[input_name] = enc["attention_mask"].astype(np.int64)
                elif input_name == "token_type_ids":
                    tti = enc.get("token_type_ids")
                    if tti is None:
                        tti = np.zeros_like(enc["input_ids"])
                    feeds[input_name] = tti.astype(np.int64)
                else:
                    if input_name in enc:
                        feeds[input_name] = enc[input_name].astype(np.int64)
                    else:
                        raise ValueError(f"ONNX 模型要求未知输入 '{input_name}'")

            output = self.sess.run([self.output_name], feeds)[0]
            if output.ndim == 3:#type: ignore
                mask = enc["attention_mask"][:, :, None].astype(np.float32)
                output = (output * mask).sum(axis=1) / mask.sum(axis=1)
            chunk_vecs.append(output[0])  #type: ignore

        # 平均池化
        avg = np.mean(chunk_vecs, axis=0)
        if normalize:
            norm = np.linalg.norm(avg)
            if norm > 1e-9:
                avg = avg / norm
        return avg.astype(np.float32)

    def close(self):
        self.sess = None

class SentenceTransformerEncoder(BaseEncoder):
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        embs = self.model.encode( #type: ignore
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=normalize
        )
        return embs.astype(np.float32)
    
    def close(self):
        self.model = None
