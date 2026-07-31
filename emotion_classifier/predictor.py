import os
import json
import asyncio
import hashlib
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

from .model import ModelConfig, ONNXEncoder


class AsyncModelDownloader:
    """异步下载模型文件，带进度条和 SHA256 校验。"""

    def __init__(self, model_dir: Path, timeout: float = 60.0):
        self.model_dir = Path(model_dir)
        self.timeout = timeout

    async def _download_one(self, client, filename: str, filepath: Path, verify: bool = True):
        url = ModelConfig.download_url(filename)
        print(f"\r  ↓ 下载 {filename} 从 {url}", file=sys.stderr)
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

            # SHA256 校验
            expected = ModelConfig.CHECKSUMS.get(filename)
            if expected:
                actual = self._sha256(filepath)
                actual_full = f"sha256:{actual}"
                if actual_full != expected:
                    filepath.unlink(missing_ok=True)
                    raise ValueError(
                        f"{filename} 校验失败: "
                        f"expected={expected[:16]}... got={actual_full[:16]}..."
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

        # 过滤已存在且校验通过的
        to_download = []
        for f in filenames:
            fp = self.model_dir / f
            if fp.exists() and ModelConfig.CHECKSUMS.get(f) is None:
                continue  # 无校验码且已存在 → 跳过
            elif fp.exists():
                if self._sha256(fp) == ModelConfig.CHECKSUMS[f]:
                    continue  # 校验通过 → 跳过
            to_download.append(f)

        if not to_download:
            print(f"  ✅ 所有模型文件已就绪 ({self.model_dir})")
            return paths

        print(f"  ⏬ 需要下载 {len(to_download)} 个文件:")
        for f in to_download:
            print(f"     • {f}")

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
        """同步包装：在同步代码中调用异步下载。"""
        downloader = AsyncModelDownloader(model_dir)
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, downloader.download(filenames)).result()
        except RuntimeError:
            return asyncio.run(downloader.download(filenames))


def _priority_sort(files: List[str]) -> List[str]:
    """按优先级排序：原始 > fp16 > int8/quant > 其他。"""
    def _pri(name: str) -> int:
        n = name.lower()
        if n == "emotion_classifier.onnx":
            return 0
        elif "fp16" in n:
            return 1
        elif "int8" in n or "quant" in n:
            return 2
        else:
            return 3
    return sorted(files, key=_pri)


def find_classifier_onnx(model_dir: Path) -> Optional[Path]:
    """
    在 model_dir 中按优先级查找分类头 ONNX 文件。
    匹配: emotion_classifier.onnx > *_fp16.onnx > *_int8.onnx > *_*.onnx

    Returns
    -------
    Optional[Path]  找到返回路径，否则 None
    """
    if not model_dir.exists():
        return None

    # 1. 精确匹配（最高优先级）
    exact = model_dir / "emotion_classifier.onnx"
    if exact.exists():
        return exact

    # 2. 通配符匹配（按优先级排序）
    import glob
    patterns = ModelConfig.CLASSIFIER_PATTERNS
    all_matches = []
    for pat in patterns:
        matches = glob.glob(str(model_dir / pat))
        all_matches.extend(matches)

    if not all_matches:
        return None

    all_matches = _priority_sort(all_matches)
    return Path(all_matches[0])


def find_encoder_onnx(model_dir: Path) -> Optional[Path]:
    """
    在 model_dir 中查找编码器 ONNX 文件。
    优先使用配置中指定的 ENCODER_ONNX，否则通配符匹配 model_*.onnx。
    """
    if not model_dir.exists():
        return None

    # 1. 精确匹配配置的编码器
    enc = model_dir / ModelConfig.ENCODER_ONNX
    if enc.exists():
        return enc

    # 2. 通配符匹配
    import glob
    matches = glob.glob(str(model_dir / "model_*.onnx"))
    if matches:
        return Path(sorted(matches)[0])

    return None


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
        - 自动匹配 emotion_classifier*.onnx（支持 fp16/int8 变体）
        - 首次使用自动从 GitHub 下载
        - 存储于 ~/emotion_classifier/models/
        - 调用 clf.update_models() 强制更新
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        backend: str = "auto",
        encoder_path: Optional[str] = None,
        auto_download: bool = False,
    ):
        self.model_dir = Path(model_dir) if model_dir else ModelConfig.default_model_dir()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.auto_download = auto_download
        self.encoder_path = encoder_path
    
    def load_label_map(self):
        label_map_path = self.model_dir / "label_map.json"
        with open(label_map_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
        self.label2id = mapping["label2id"]
        self.num_labels = mapping["num_labels"]

    
    def init(self):
        # ── 解析需要的文件 ──
        needed = self._resolve_needed_files(self.backend)
        missing = [f for f in needed if not (self.model_dir / f).exists()]

        if missing:
            if self.auto_download:
                print(f"[初始化] 检测到 {len(missing)} 个模型文件缺失，开始下载...")
                # AsyncModelDownloader.download_sync(needed, self.model_dir)
            else:
                raise FileNotFoundError(
                    f"模型文件缺失: {missing}\n"
                    f"请设置 auto_download=True 或手动放置到 {self.model_dir}"
                )
        self.load_label_map()
        self._init_backend(self.backend)
        self._init_encoder(self.encoder_path)

    def _init_encoder(self, encoder_path: Optional[str]):
        """初始化文本编码器（ONNX 或 SentenceTransformer）。"""
        if encoder_path:
            enc_path = Path(encoder_path)
            if enc_path.exists() and enc_path.suffix == ".onnx":
                self._load_onnx_encoder(enc_path)
                return
            else:
                # 当作 SentenceTransformer 名称处理
                self._load_st_encoder(str(encoder_path))
                return

        # 2. 自动查找 ONNX 编码器
        enc_onnx = find_encoder_onnx(self.model_dir)
        if enc_onnx:
            self._load_onnx_encoder(enc_onnx)
            return

        # 3. 回退到 SentenceTransformer
        self._load_st_encoder(ModelConfig.ENCODER_TOKENIZER)

    def _load_onnx_encoder(self, onnx_path: Path):
        print(f"[编码器] 加载量化 ONNX: {onnx_path.name}")
        # 查找配套 tokenizer 目录
        tokenizer_dir = None
        for candidate in [self.model_dir, self.model_dir / ModelConfig.ENCODER_TOKENIZER_DIR]:
            if candidate.exists() and (candidate / "tokenizer.json").exists():
                tokenizer_dir = str(candidate)
                break

        self.encoder = ONNXEncoder(
            onnx_path=str(onnx_path),
            tokenizer_dir=tokenizer_dir or str(self.model_dir / ModelConfig.ENCODER_TOKENIZER_DIR),
            tokenizer_name=f"sentence-transformers/{ModelConfig.ENCODER_TOKENIZER}"
        )
        self._encode_method = "onnx"

    def _load_st_encoder(self, model_name: str):
        print(f"[编码器] 加载 SentenceTransformer: {model_name}")
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name)
        self._encode_method = "st"

    def _resolve_needed_files(self, backend: str) -> List[str]:
        """
        根据后端类型和本地已有文件，确定需要下载的文件列表。
        分类头 ONNX 使用通配符匹配 emotion_classifier*.onnx。
        """
        if backend == "onnx":
            return list(ModelConfig.ONNX_FILES)
        elif backend == "pytorch":
            return list(ModelConfig.PT_FILES)
        else:
            # auto：检测本地已有文件
            cls_onnx = find_classifier_onnx(self.model_dir)
            if cls_onnx:
                # 有 ONNX 分类头 → 下载 ONNX 套件
                return list(ModelConfig.ONNX_FILES)
            elif (self.model_dir / "emotion_classify.pt").exists():
                return list(ModelConfig.PT_FILES)
            else:
                # 都没下载过 → 默认下载 ONNX
                return list(ModelConfig.ONNX_FILES)

    def _init_backend(self, backend: str):
        if backend == "auto":
            cls_onnx = find_classifier_onnx(self.model_dir)
            if cls_onnx:
                backend = "onnx"
                self._classifier_onnx_path = cls_onnx
                print(f"[后端] 自动选择 ONNX: {cls_onnx.name}")
            elif (self.model_dir / "emotion_classify.pt").exists():
                backend = "pytorch"
            else:
                # 都没下载 → 默认 ONNX
                backend = "onnx"
                self._classifier_onnx_path = self.model_dir / ModelConfig.CLASSIFIER_ONNX_DEFAULT

        self.backend = backend

        if backend == "onnx":
            self._init_onnx()
        else:
            self._init_pytorch()

    def _init_onnx(self):
        import onnxruntime as ort

        # 通配符查找分类头
        cls_path = getattr(self, '_classifier_onnx_path', None)
        if cls_path is None:
            cls_path = find_classifier_onnx(self.model_dir)
        if cls_path is None:
            cls_path = self.model_dir / ModelConfig.CLASSIFIER_ONNX_DEFAULT

        print(f"[后端] 加载分类头 ONNX: {cls_path.name}")

        providers = ["CPUExecutionProvider"]
        try:
            import torch
            if torch.cuda.is_available():
                providers.insert(0, "CUDAExecutionProvider")
        except ImportError:
            pass

        self.sess = ort.InferenceSession(str(cls_path), providers=providers)
        self._input_name = self.sess.get_inputs()[0].name
        self._output_names = [self.sess.get_outputs()[0].name] # type: ignore
        self._input_dtype = self.sess.get_inputs()[0].type

    def _init_pytorch(self):
        import torch
        from emotion_classifier.model import EmotionClassifierNet

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionClassifierNet(
            input_dim=384,
            num_classes=self.num_labels,
            hidden_dim=256,
            dropout=0.5,
        ).to(self.device)

        state = torch.load(
            self.model_dir / "emotion_classify.pt",
            map_location=self.device
        )
        self.model.load_state_dict(state)
        self.model.eval()

    def encode(self, texts: List[str]|str):
        if isinstance(texts, str):
            texts = [texts]

        if self._encode_method == "onnx":
            return self.encoder.encode(texts)
        else:
            embs = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False) # type: ignore
            return embs.astype(np.float32)

    def predict(self, texts: List[str]|str, top_k: int = 3) -> List[List[Tuple[str, float]]]:
        """
        情感分类预测。

        Returns
        -------
        list[list[tuple[str, float]]]
            每条文本的前 top_k 个 (标签, 概率)
        """
        embeddings = self.encode(texts)
        if self._input_dtype == "tensor(float16)":
            embeddings = embeddings.astype(np.float16) # type: ignore
        else:
            embeddings = embeddings.astype(np.float32) # type: ignore

        if self.backend == "onnx":
            logits = self.sess.run(self._output_names, {self._input_name: embeddings})[0]
        else:
            import torch
            with torch.no_grad():
                logits = self.model(
                    torch.from_numpy(embeddings).to(self.device)
                ).cpu().numpy()

        # 数值稳定 softmax
        logits = logits.astype(np.float32) - logits.astype(np.float32).max(axis=1, keepdims=True) # type: ignore
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

    def list_available_models(self) -> Dict[str, List[str]]:
        """
        列出 model_dir 中所有可用的模型文件。
        用于调试和信息展示。
        """
        import glob
        result = {"classifier_onnx": [], "encoder_onnx": [], "pytorch": []}

        # 分类头
        for f in glob.glob(str(self.model_dir / "emotion_classifier*.onnx")):
            result["classifier_onnx"].append(Path(f).name)

        # 编码器
        for f in glob.glob(str(self.model_dir / "model_*.onnx")):
            result["encoder_onnx"].append(Path(f).name)

        # PyTorch
        pt = self.model_dir / "emotion_classify.pt"
        if pt.exists():
            result["pytorch"].append(pt.name)

        return result

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
            "available_models": self.list_available_models(),
            "files": {},
        }

        # 检查关键文件
        key_files = ["label_map.json"]
        # 加上匹配到的分类头
        cls = find_classifier_onnx(self.model_dir)
        if cls:
            key_files.append(cls.name)
        else:
            key_files.append("emotion_classify.pt")

        for f in key_files:
            fp = self.model_dir / f
            info["files"][f] = {
                "exists": fp.exists(),
                "size_mb": round(fp.stat().st_size / 1e6, 2) if fp.exists() else 0,
            }
        return info

    def __repr__(self):
        return (
            f"EmotionClassifier(backend='{self.backend}', "
            f"encoder='{self._encode_method}', "
            f"labels={self.num_labels}, "
            f"dir='{self.model_dir}')"
        )