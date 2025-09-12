"""qudas.core.base

旧 API 互換性のために残されているモジュールです。
各基底クラスの実装は個別ファイルに分割されました。
今後は ``qudas.core.input_base`` などを直接 import してください。
"""

from __future__ import annotations

from .input_base import QdInputBase
from .output_base import QdOutputBase
from .executor_base import QdExecutorBase

# 旧クラス名のエイリアス ----------------------------------------------------
QdInBase = QdInputBase
QdOutBase = QdOutputBase
QdExecBase = QdExecutorBase

__all__ = [
    "QdInputBase",
    "QdOutputBase",
    "QdExecutorBase",
    "QdInBase",
    "QdOutBase",
    "QdExecBase",
]
