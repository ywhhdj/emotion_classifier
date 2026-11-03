"""Rust 引擎单例与后端选择（内部实现细节，不对外暴露）。

后端优先级：rust > onnx > pytorch，可通过 EMOTION_BACKEND 覆盖。
"""

import os
import threading
from pathlib import Path
from typing import Optional

from .exceptions import BackendError, ModelLoadError


_lock = threading.Lock()
_engine_cache = {}


def _has_rust_core() -> bool:
    try:
        from . import _core  # noqa: F401
        return True
    except Exception:
        return False


def select_backend(preferred: Optional[str] = None) -> str:
    """返回实际可用的后端名。"""
    pref = (preferred or os.environ.get("EMOTION_BACKEND") or "rust").lower()
    order = {"rust": ["rust", "onnx", "pytorch"],
             "onnx": ["onnx", "pytorch"],
             "pytorch": ["pytorch"]}.get(pref, ["onnx", "pytorch"])
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


def get_engine(model_dir: str | Path, num_threads: int = 4, max_batch: int = 32,
               max_wait_ms: int = 2, backend: str = "candle", precision: str = "fp16"):
    """按 (model_dir, 配置) 复用 Rust 引擎实例（权重 mmap 共享）。"""
    key = (str(model_dir), num_threads, max_batch, max_wait_ms, backend, precision)
    with _lock:
        eng = _engine_cache.get(key)
        if eng is not None:
            return eng
        try:
            from . import _core
        except ImportError as e:
            raise ModelLoadError(
                "Rust 内核不可用，请安装带内核的 wheel 或设置 EMOTION_BACKEND=onnx"
            ) from e
        try:
            eng = _core.EmotionEngine(
                model_dir=str(model_dir), num_threads=num_threads, max_batch=max_batch,
                max_wait_ms=max_wait_ms, backend=backend, precision=precision,
            )
        except Exception as e:
            raise ModelLoadError(f"Rust 引擎初始化失败: {e}") from e
        _engine_cache[key] = eng
        return eng


def release_engines():
    with _lock:
        for eng in _engine_cache.values():
            try:
                eng.close()
            except Exception:
                pass
        _engine_cache.clear()
