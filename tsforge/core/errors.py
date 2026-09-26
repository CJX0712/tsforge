"""errors — 分层错误码（E1xx 体系，与 recforge 同构）。

使用：抛出具体子类；上层按 code 区分处理。
"""

from __future__ import annotations


class TSForgeError(Exception):
    """基类。"""

    code = "E000"


class ConfigError(TSForgeError):
    code = "E100"


class DataError(TSForgeError):
    code = "E200"


class ModelError(TSForgeError):
    code = "E300"


class EvalError(TSForgeError):
    code = "E400"
