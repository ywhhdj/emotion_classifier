"""异常层级（Rust 内核错误会被映射到这里）。"""


class EmotionClassifierError(Exception):
    pass


class ModelLoadError(EmotionClassifierError):
    pass


class InvalidInputError(EmotionClassifierError):
    pass


class BackendError(EmotionClassifierError):
    pass


class TokenizerError(EmotionClassifierError):
    pass
