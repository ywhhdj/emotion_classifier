"""v1 -> v2 兼容层。

保持旧代码零改动可用：
  * 旧模型目录（model_quint8_avx2.onnx + 384 维分类头）自动识别并走旧路径
  * 旧返回结构 [(label, score), ...] 不变
  * 新增能力通过新参数开关，不破坏旧签名
"""

from typing import List, Optional, Tuple, Union

import numpy as np

from .predictor import EmotionClassifier as _New
from .exceptions import InvalidInputError


class LegacyClassifier(_New):
    """v1 行为兼容外壳。

    差异仅在于：
      * 默认不加 lang token（v1 输入是裸文本）
      * 默认严格走 ONNX（不强制 Rust 内核）
    """

    def __init__(self, *args, use_lang_token: bool = False, **kwargs):
        kwargs.setdefault("backend", "onnx")
        super().__init__(*args, use_lang_token=use_lang_token, **kwargs)

    def predict_one(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """v1 常用的「单条」入口，直接返回 [(label, score), ...]（不是嵌套 list）。"""
        if not isinstance(text, str):
            raise InvalidInputError("predict_one 只接受单个字符串")
        return self.predict(text, top_k=top_k)[0]

    def predict_proba_matrix(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.predict_proba(texts)


def is_legacy_model_dir(model_dir) -> bool:
    """旧产物判定：存在 model_quint8_avx2.onnx 或分类头输入为 384 维。"""
    from pathlib import Path
    p = Path(model_dir)
    if (p / "model_quint8_avx2.onnx").exists():
        return True
    try:
        import onnxruntime as ort
        from .models.config import ModelRepository
        cls = ModelRepository(p).find_classifier_onnx()
        if cls is None:
            return False
        sess = ort.InferenceSession(str(cls), providers=["CPUExecutionProvider"])
        ins = {i.name: i for i in sess.get_inputs()}
        if "input_ids" in ins:
            return False
        shape = list(ins.values())[0].shape
        return len(shape) == 2 and isinstance(shape[-1], int) and shape[-1] == 384
    except Exception:
        return False


def load_for_model_dir(model_dir: Optional[str] = None, **kwargs):
    """按模型目录自动选择新版 / 兼容版实例。"""
    from .models.config import ModelConfig
    d = model_dir or str(ModelConfig.default_model_dir())
    if is_legacy_model_dir(d):
        return LegacyClassifier(model_dir=d, **kwargs)
    return _New(model_dir=d, **kwargs)
