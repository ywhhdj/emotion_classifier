import os
import json
import asyncio
import hashlib
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
from .model import ModelConfig

class AsyncModelDownloader:
    def __init__(self, model_dir: Path, timeout: float = 60.0):
        self.model_dir = Path(model_dir)
        self.timeout = timeout

    async def _download_one(self, client, filename: str, filepath: Path, verify: bool = True):
        """下载单个文件，带进度回调和校验。"""
        url = ModelConfig.download_url(filename)
        if filepath.exists() and not verify:
            return filepath
        tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
        try:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 64 * 1024  # 64KB

                with open(tmp_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            bar_len = int(pct / 5)
                            bar = "█" * bar_len + "░" * (20 - bar_len)
                            sys.stderr.write(
                                f"\r  ↓ {filename:30s} [{bar}] {pct:5.1f}%"
                            )
                            sys.stderr.flush()
            sys.stderr.write("\n")
            os.replace(tmp_path, filepath)
            expected = ModelConfig.CHECKSUMS.get(filename)
            if expected:
                actual = self._sha256(filepath)
                if actual != expected:
                    filepath.unlink(missing_ok=True)
                    raise ValueError(
                        f"{filename} 校验失败: expected={expected[:12]}... "
                        f"got={actual[:12]}..."
                    )
            return filepath
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"下载 {filename} 失败: {e}") from e

    @staticmethod
    def _sha256(filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    async def download(self, filenames: List[str]) -> List[Path]:
        import httpx

        self.model_dir.mkdir(parents=True, exist_ok=True)
        paths = [self.model_dir / f for f in filenames]

        # 过滤已存在且无需更新的
        to_download = []
        for f in filenames:
            fp = self.model_dir / f
            if fp.exists() and ModelConfig.CHECKSUMS.get(f) is None:
                # 无校验码且已存在 → 跳过
                continue
            elif fp.exists():
                # 有校验码 → 验证
                if self._sha256(fp) == ModelConfig.CHECKSUMS[f]:
                    continue
            to_download.append(f)

        if not to_download:
            print(f"  ✅ 所有模型文件已就绪 ({self.model_dir})")
            return paths

        print(f"  ⏬ 需要下载 {len(to_download)} 个文件:")
        for f in to_download:
            print(f"     • {f}")

        # 并发下载
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
        ) as client:
            await asyncio.gather(*[
                self._download_one(client, f, self.model_dir / f)
                for f in to_download
            ])

        print(f"  ✅ 下载完成 → {self.model_dir}")
        return [self.model_dir / f for f in filenames]

    @staticmethod
    def download_sync(filenames: List[str], model_dir: Path) -> List[Path]:
        downloader = AsyncModelDownloader(model_dir)
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, downloader.download(filenames)).result()
        except RuntimeError:
            return asyncio.run(downloader.download(filenames))

class EmotionClassifier:
    """
    多语言情感分类器（19 种情感，中英日）。

    用法:
        clf = EmotionClassifier()              # 自动选择后端
        clf = EmotionClassifier(backend='onnx')
        clf = EmotionClassifier(backend='pytorch')

        result = clf.predict("脸颊泛红，偷偷瞄了你一眼")
        # [('害羞', 0.80), ('生气', 0.12), ('高兴', 0.03)]

    模型文件管理:
        - 首次使用自动从 GitHub 下载
        - 存储于 ~/.emotion_classifier/models/（可用 EMOTION_MODEL_DIR 覆盖）
        - 调用 clf.update_models() 强制更新
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        backend: str = "auto",
        encoder_path: Optional[str] = None,
        auto_download: bool = True,
    ):
        """
        Parameters
        ----------
        model_dir : str | None
            模型文件目录，默认 ~/.emotion_classifier/models/
        backend : str
            'onnx' | 'pytorch' | 'auto'
        encoder_path : str | None
            SentenceTransformer 路径或名称
        auto_download : bool
            本地缺失时是否自动下载，默认 True
        """
        self.model_dir = Path(model_dir) if model_dir else ModelConfig.default_model_dir()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.auto_download = auto_download

        needed = self._resolve_needed_files(backend)
        missing = [f for f in needed if not (self.model_dir / f).exists()]

        if missing:
            if auto_download:
                print(f"[初始化] 检测到 {len(missing)} 个模型文件缺失，开始下载...")
                AsyncModelDownloader.download_sync(needed, self.model_dir)
            else:
                raise FileNotFoundError(
                    f"模型文件缺失: {missing}\n"
                    f"请设置 auto_download=True 或手动放置到 {self.model_dir}"
                )

        label_map_path = self.model_dir / "label_map.json"
        with open(label_map_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
        self.label2id = mapping["label2id"]
        self.num_labels = mapping["num_labels"]

        self._init_backend(backend)
        try:
            from sentence_transformers import SentenceTransformer
            model_name = encoder_path or "paraphrase-multilingual-MiniLM-L12-v2"
            self.encoder = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("请先安装paraphrase-multilingual-MiniLM-L12-v2模型")

    def _resolve_needed_files(self, backend: str) -> List[str]:
        if backend == "onnx":
            return list(ModelConfig.ONNX_FILES)
        elif backend == "pytorch":
            return list(ModelConfig.PT_FILES)
        else:
            # 优先 ONNX（更快、更轻）
            if (self.model_dir / "emotion_classifier.onnx").exists():
                return list(ModelConfig.ONNX_FILES)
            elif (self.model_dir / "emotion_classify.pt").exists():
                return list(ModelConfig.PT_FILES)
            else:
                # 都没下载过 → 默认下载 ONNX
                return list(ModelConfig.ONNX_FILES)

    def _init_backend(self, backend: str):
        """初始化选定的推理后端。"""
        if backend == "auto":
            if (self.model_dir / "emotion_classifier.onnx").exists():
                backend = "onnx"
            else:
                backend = "pytorch"
        self.backend = backend

        if backend == "onnx":
            self._init_onnx()
        else:
            self._init_pytorch()

    def _init_onnx(self):
        import onnxruntime as ort
        onnx_path = str(self.model_dir / "emotion_classifier.onnx")
        # 自动选择最佳 provider
        providers = ["CPUExecutionProvider"]
        from torch import cuda
        if cuda.is_available():
            providers.insert(0, "CUDAExecutionProvider")
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self.sess.get_inputs()[0].name
        self._output_names = [self.sess.get_outputs()[0].name]

    def _init_pytorch(self):
        import torch
        from model import EmotionClassifierNet
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionClassifierNet(
            input_dim=384,
            num_classes=self.num_labels,
            hidden_dim=256,
            dropout=0.5,
        ).to(self.device)
        state = torch.load(
            self.model_dir / "emotion_classify.pt", map_location=self.device
        )
        self.model.load_state_dict(state)
        self.model.eval()

    # ────────────────────────────────────────
    # 公开 API
    # ────────────────────────────────────────
    def encode(self, texts):
        """文本 → 384 维向量。"""
        if isinstance(texts, str):
            texts = [texts]
        embs = self.encoder.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embs.astype(np.float32)

    def predict(self, texts, top_k: int = 3) -> List[List[Tuple[str, float]]]:
        """
        情感分类预测。

        Returns
        -------
        list[list[tuple[str, float]]]
            每条文本的前 top_k 个 (标签, 概率)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.encode(texts)

        if self.backend == "onnx":
            logits = self.sess.run(self._output_names, {self._input_name: embeddings})[0]
        else:
            import torch
            with torch.no_grad():
                logits = self.model(
                    torch.from_numpy(embeddings).to(self.device)
                ).cpu().numpy()

        # 数值稳定 softmax
        logits = logits - logits.max(axis=1, keepdims=True) # type: ignore
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

    def predict_label(self, text: str, top_k: int = 1) -> List[Tuple[str, float]]:
        return self.predict(text, top_k=top_k)[0]

    def get_labels(self) -> List[str]:
        return list(self.label2id.keys())

    def update_models(self):
        print(f"[更新] 检查并重新下载模型文件...")
        needed = self._resolve_needed_files(self.backend)
        for f in needed:
            fp = self.model_dir / f
            if fp.exists():
                fp.unlink()
                print(f"  已删除旧文件: {f}")
        AsyncModelDownloader.download_sync(needed, self.model_dir)
        self._init_backend(self.backend)
        print("[更新] 完成，模型已刷新")

    def model_info(self) -> Dict:
        info = {
            "backend": self.backend,
            "model_dir": str(self.model_dir),
            "num_labels": self.num_labels,
            "labels": self.get_labels(),
            "files": {},
        }
        for f in ["emotion_classify.pt", "emotion_classifier.onnx", "label_map.json"]:
            fp = self.model_dir / f
            info["files"][f] = {
                "exists": fp.exists(),
                "size_mb": round(fp.stat().st_size / 1e6, 2) if fp.exists() else 0,
            }
        return info

    def __repr__(self):
        return (
            f"EmotionClassifier(backend='{self.backend}', "
            f"labels={self.num_labels}, "
            f"dir='{self.model_dir}')"
        )