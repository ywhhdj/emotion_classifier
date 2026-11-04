"""进程内唯一推理实例

1. 全局只有一个 `ModelHandle`。`get_engine()` 在配置相同时直接复用，
   配置不同时**报错**而不是再初始化一个引擎；
2. 调用方可用 `attach_model()` 注入已加载的 session / Rust engine / ONNX 字节，
   生命周期由调用方决定（`owns=False` 时库不主动 close）；
3. 只有显式调用 `release_model()` / `replace=True` 才会销毁并允许重新初始化。

`EmotionClassifier.close()` 只解除自身引用，不会销毁共享实例；
"""

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Optional

from .exceptions import BackendError, InvalidInputError, ModelLoadError
from .models.loaded import LoadedModel
from .models.config import normalize_precision

_LOCK = threading.RLock()
_HANDLE: Optional["ModelHandle"] = None


class ModelHandle:
    """单例容器：记录配置指纹与所有权。"""

    __slots__ = ("model", "key")

    def __init__(self, model: LoadedModel, key: tuple):
        self.model = model
        self.key = key

    def close(self) -> None:
        m = self.model
        if not m.owns or m.session is None:
            return
        try:
            closer = getattr(m.session, "close", None)
            if callable(closer):
                closer()
        except Exception:
            pass
        m.session = None


def _has_rust_core() -> bool:
    try:
        from . import _core  # type: ignore
        return True
    except Exception:
        return False


def _release_rust_global() -> None:
    """清除 Rust 侧全局引擎。

    `replace=True` 换档时，Python 侧 `_HANDLE` 被清掉并不等于 Rust 全局实例
    被释放；`global_engine()` 会用旧配置指纹拒绝新引擎。必须在创建新引擎前
    显式清一次，否则 Rust 后端的换档/热更新必然失败。
    """
    try:
        from . import _core # type: ignore
        releaser = getattr(_core, "release_engine", None)
        if callable(releaser):
            releaser()
    except Exception:
        pass


def select_backend(preferred: Optional[str] = None) -> str:
    """返回实际可用的后端名（rust > onnx > pytorch）。"""
    pref = (preferred or os.environ.get("EMOTION_BACKEND") or "rust").lower()
    order = {"rust": ["rust", "onnx", "pytorch"],
             "onnx": ["onnx", "pytorch"],
             "pytorch": ["pytorch"]}.get(pref, ["rust", "onnx", "pytorch"])
    for cand in order:
        if cand == "rust":
            if _has_rust_core():
                return "rust"
        elif cand == "onnx":
            try:
                import onnxruntime  # noqa: F401
                return "onnx"
            except Exception:
                continue
        else:
            try:
                import torch  # noqa: F401
                return "pytorch"
            except Exception:
                continue
    raise BackendError("没有可用的推理后端（需要 Rust 内核或 onnxruntime 或 torch）")


def current_model() -> Optional[LoadedModel]:
    """返回当前唯一实例（不触发初始化）。"""
    with _LOCK:
        return _HANDLE.model if _HANDLE is not None else None


def _ort_session(source: Any, num_threads: int = 4):
    """创建 ORT session。

    优化级别对齐训练侧 `train/train.py::_ort_session`：默认 BASIC。
    ORT 的 SkipLayerNormFusion 在 XLM-R 整图上会产出 shape 不匹配的
    SkipLayerNormalization 节点，ENABLE_ALL 会触发该融合（v0.1 的隐性 bug）。
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    level = os.environ.get("EMO_ORT_OPT_LEVEL", "basic").lower()
    opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL if level == "all"
        else ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    opts.log_severity_level = 3
    opts.intra_op_num_threads = max(1, int(num_threads))
    opts.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    if os.environ.get("EMOTION_DEVICE", "").lower() == "cuda" and \
            "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(source, sess_options=opts, providers=providers)


def _is_ort_session(obj: Any) -> bool:
    return hasattr(obj, "get_inputs") and hasattr(obj, "run")


def _is_rust_engine(obj: Any) -> bool:
    return hasattr(obj, "forward") and hasattr(obj, "num_labels")


def _coerce(
    model: Any, *, 
    precision: str,
    backend: str,
    owns: bool,
    source: str,
    num_threads: int = 4,
    tokenizer: Any = None, label_map: Any = None, calib: Any = None,
    model_path: Optional[str] = None
) -> LoadedModel:
    """把各种「已加载模型」入参归一成 LoadedModel。"""
    if isinstance(model, LoadedModel):
        m = model
        m.precision = normalize_precision(m.precision)
        if tokenizer is not None:
            m.tokenizer = tokenizer
        if label_map is not None:
            m.label_map = label_map
        if calib is not None:
            m.calib = calib
        if model_path is not None:
            m.model_path = str(model_path)
        if m.session is None and m.model_path:
            # 只给了路径：session 由库创建，因此只有库能安全释放它
            m.session = _ort_session(m.model_path, num_threads)
            m.owns = True
        if m.session is None:
            raise InvalidInputError("LoadedModel.session 为空，且没有可用的 model_path")
        # 尊重调用方显式声明的 backend（尤其 pytorch / 自定义 Rust 句柄），
        # 只在值不合法时才按 session 形态兜底判定
        if m.backend not in ("onnx", "rust", "pytorch"):
            m.backend = "rust" if _is_rust_engine(m.session) else "onnx"
        return m

    p = normalize_precision(precision)
    if isinstance(model, (bytes, bytearray, memoryview)):
        raw = bytes(model)
        if not raw:
            raise InvalidInputError("模型字节为空")
        return LoadedModel(session=_ort_session(raw, num_threads),
                           tokenizer=tokenizer, label_map=label_map, calib=calib,
                           precision=p, backend="onnx", owns=True,
                           source="memory",
                           meta={"fingerprint": hashlib.sha256(raw).hexdigest()[:16]})

    if isinstance(model, (str, Path)):
        path = Path(model)
        if not path.exists():
            raise ModelLoadError(f"模型文件不存在: {path}")
        return LoadedModel(session=_ort_session(str(path), num_threads),
                           tokenizer=tokenizer, label_map=label_map, calib=calib,
                           precision=precision_of_path(path), backend="onnx",
                           owns=True, source="file", meta={},
                           model_path=str(path))

    if _is_rust_engine(model):
        return LoadedModel(session=model, tokenizer=tokenizer, label_map=label_map,
                           calib=calib, precision=p, backend="rust", owns=owns,
                           source=source)

    if _is_ort_session(model):
        return LoadedModel(session=model, tokenizer=tokenizer, label_map=label_map,
                           calib=calib, precision=p, backend="onnx", owns=owns,
                           source=source)

    raise InvalidInputError(
        "model 必须是 LoadedModel / onnxruntime.InferenceSession / "
        "_core.EmotionEngine / ONNX 字节 / .onnx 路径"
    )


def precision_of_path(path: Path) -> str:
    from .models.config import precision_of
    return precision_of(path)


def _external_key(m: LoadedModel) -> tuple:
    session = m.session
    fingerprint = ""
    if isinstance(m.meta.get("fingerprint"), str):
        fingerprint = m.meta["fingerprint"]
    elif m.model_path:
        fingerprint = str(Path(m.model_path).resolve())
    return ("external", m.backend, m.precision, type(session).__name__,
            fingerprint, id(session))


def attach_model(
    model: Any, *, 
    precision: str = "int8", 
    backend: str = "onnx",
    owns: bool = False,
    replace: bool = False,
    tokenizer: Any = None,
    label_map: Any = None,
    calib: Any = None,
    model_path: Optional[str] = None,
    num_threads: int = 4
) -> LoadedModel:
    """登记一个由调用方管理的已加载模型为进程内唯一实例。

    `replace=True` 会先释放旧实例（用于热更新 / 切换档位）。
    """
    global _HANDLE
    # 冲突预检必须在 _coerce 之前：注入字节/路径时 _coerce 会真的创建一个
    # ORT session，旧实现「先建第二份模型再报错」既违背单一实例，又泄漏会话。
    candidate = model.session if isinstance(model, LoadedModel) else model
    with _LOCK:
        if _HANDLE is not None:
            if _HANDLE.model.session is candidate:
                return _HANDLE.model
            if not replace:
                raise BackendError(
                    "进程内只允许一个模型实例；已存在 "
                    f"{_HANDLE.model.source}/{_HANDLE.model.precision}。"
                    "如需替换请先调用 release_model() 或传 replace=True"
                )
            if _HANDLE.model.backend == "rust":
                _release_rust_global()
            _HANDLE.close()
            _HANDLE = None
        loaded = _coerce(
            model,
            precision=precision,
            backend=backend,
            owns=owns,
            source="external",
            num_threads=num_threads,
            tokenizer=tokenizer,
            label_map=label_map,
            calib=calib,
            model_path=model_path
        )
        _HANDLE = ModelHandle(loaded, _external_key(loaded))
        return loaded


def _internal_key(**cfg) -> tuple:
    return (
        "internal", str(cfg["model_dir"]), cfg["precision"], cfg["backend"],
        int(cfg["num_threads"]), int(cfg["max_batch"]), int(cfg["max_wait_ms"]),
        int(cfg["max_seq_len"]), bool(cfg["use_lang_token"]),
        str(cfg.get("model_path") or ""),
    )


def get_engine(
    *,
    model_dir: "str | Path",
    tokenizer_dir: Optional["str | Path"] = None,
    precision: str = "fp32",
    backend: str = "rust",
    num_threads: int = 4,
    max_batch: int = 32,
    max_wait_ms: int = 2,
    max_seq_len: int = 64,
    use_lang_token: bool = False,
    model_path: Optional[str] = None,
    replace: bool = False
) -> LoadedModel:
    """获取（且只初始化一次）进程内唯一的内部推理实例。

    backend="rust" 时构造 PyO3 内核（加载三档 ONNX 整图）；
    backend="onnx" 时在 Python 侧创建 ORT session。两者都纳入同一单例。
    """
    global _HANDLE
    p = normalize_precision(precision)
    resolved_path = str(Path(model_path).resolve()) if model_path else ""
    cfg = dict(
        model_dir=Path(model_dir).resolve(),
        tokenizer_dir=Path(tokenizer_dir).resolve() if tokenizer_dir else None,
        precision=p,
        backend=backend,
        num_threads=num_threads,
        max_batch=max_batch,
        max_wait_ms=max_wait_ms,
        max_seq_len=max_seq_len,
        use_lang_token=use_lang_token,
        model_path=resolved_path
    )
    key = _internal_key(**cfg)

    with _LOCK:
        if _HANDLE is not None:
            if _HANDLE.key == key:
                return _HANDLE.model          # 复用，绝不重复初始化
            if _HANDLE.key[0] == "external":
                if not replace:
                    raise BackendError(
                        "已注入外部模型，进程内只允许一个实例；"
                        "如需切换请先 release_model()（或 replace=True）"
                    )
            elif not replace:
                raise BackendError(
                    "进程内只允许一个模型实例；当前实例配置为 "
                    f"{_HANDLE.key}，请求配置为 {key}。"
                    "如需切换请先调用 release_model()（或 replace=True）"
                )
            _HANDLE.close()
            _HANDLE = None
        if replace:
            # Python 句柄清了还不够：Rust 全局实例会按旧指纹拒绝新引擎
            _release_rust_global()

        if backend == "rust":
            try:
                from . import _core # type: ignore
            except ImportError as exc:
                raise ModelLoadError(
                    "Rust 内核不可用，请安装带内核的 wheel 或设置 "
                    "EMOTION_BACKEND=onnx 回退到 ONNX Runtime"
                ) from exc
            try:
                engine = _core.EmotionEngine(
                    model_dir=str(cfg["model_dir"]),
                    tokenizer_dir=str(cfg["tokenizer_dir"]) if cfg["tokenizer_dir"] else None,
                    num_threads=int(num_threads),
                    max_batch=int(max_batch),
                    max_wait_ms=int(max_wait_ms),
                    backend="onnx",
                    precision=p,
                    max_seq_len=int(max_seq_len),
                    use_lang_token=bool(use_lang_token),
                    model_path=resolved_path or None,
                    model=None,
                )
            except Exception as exc:
                raise ModelLoadError(f"Rust 引擎初始化失败: {exc}") from exc
            loaded = LoadedModel(session=engine, precision=p, backend="rust",
                                 owns=True, source="internal", meta=dict(cfg))
        elif backend == "onnx":
            from .models.config import ModelRepository
            repo = ModelRepository(cfg["model_dir"], create=False) # type: ignore
            cls = repo.find_classifier_onnx(p)
            if cls is None:
                raise ModelLoadError(
                    f"未找到 ONNX 模型: {cfg['model_dir']}（期望 "
                    f"{repo.model_file(p)}）"
                )
            loaded = LoadedModel(
                session=_ort_session(str(cls), num_threads),
                precision=p, backend="onnx", owns=True, source="internal",
                model_path=str(cls), meta=dict(cfg))
        else:
            raise BackendError(f"不支持的内部后端: {backend}")

        _HANDLE = ModelHandle(loaded, key)
        return loaded


def release_model() -> bool:
    """显式释放唯一实例（含 Rust 全局引擎）。返回是否确实释放了资源。"""
    global _HANDLE
    with _LOCK:
        if _HANDLE is None:
            # 即使没有 Python 侧句柄，也可能存在 Rust 全局实例
            _release_rust_global()
            return False
        _HANDLE.close()
        _HANDLE = None
        _release_rust_global()
        return True


def assert_compatible(
    *,
    model_dir: str | Path, 
    precision: str = "fp32",
    num_threads: int = 4,
    max_batch: int = 32,
    max_wait_ms: int = 2,
    max_seq_len: int = 64,
    use_lang_token: bool = False,
    model_path: Optional[str] = None,
    replace: bool = False,
    backends: tuple = ("rust", "onnx")
) -> None:
    """只读预检：已有唯一实例是否允许本次内部加载（绝不创建任何模型）。

    `EmotionClassifier` 的后端回退链会依次尝试 rust/onnx/pytorch；若把
    「配置冲突」当成「内核不可用」吞掉继续回退，就会绕过单例再加载一份模型。
    这里用与 `get_engine` 完全相同的指纹提前判定，冲突直接向上抛。
    """
    if replace:
        return
    p = normalize_precision(precision)
    resolved = str(Path(model_path).resolve()) if model_path else ""
    md = str(Path(model_dir).resolve())
    with _LOCK:
        if _HANDLE is None:
            return
        key = _HANDLE.key
        if key[0] != "internal":
            # PyTorch 回退实例也由本库加载；同后端复用，跨后端一律拒绝
            if _HANDLE.model.backend == "pytorch" and backends == ("pytorch",):
                return
            raise BackendError(
                "已注入外部模型，进程内只允许一个实例；"
                "如需切换请先 release_model()（或 replace=True）"
            )
        for b in backends:
            candidate = _internal_key(
                model_dir=md, precision=p, backend=b,
                num_threads=num_threads, max_batch=max_batch,
                max_wait_ms=max_wait_ms, max_seq_len=max_seq_len,
                use_lang_token=use_lang_token, model_path=resolved
            )
            if candidate == key:
                return
        raise BackendError(
            "进程内只允许一个模型实例；当前实例配置为 "
            f"{key}，请求精度 {p} / 候选后端 {backends}。"
            "如需切换请先调用 release_model()（或 replace=True）"
        )


# 兼容 v0.1 内部调用名
release_engines = release_model


def get_engine_cached(model_dir: "str | Path", num_threads: int = 4,
                      max_batch: int = 32, max_wait_ms: int = 2,
                      backend: str = "candle", precision: str = "fp16"):
    """[兼容] 旧签名：映射到新的单一实例接口。"""
    return get_engine(model_dir=model_dir, precision=precision,
                      backend="rust" if backend in ("candle", "onnx", "ort") else backend,
                      num_threads=num_threads, max_batch=max_batch,
                      max_wait_ms=max_wait_ms)
