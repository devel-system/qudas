"""qudas.core.base

旧 API 互換性のために残されているモジュールです。
各基底クラスの実装は個別ファイルに分割されました。
今後は ``qudas.core.input_base`` などを直接 import してください。
"""

from __future__ import annotations

from .input_base import QuDataInputBase
from .output_base import QuDataOutputBase
from .executor_base import QuExecutorBase

# 旧クラス名のエイリアス ----------------------------------------------------
QdInBase = QuDataInputBase
QdOutBase = QuDataOutputBase
QdExecBase = QuExecutorBase

__all__ = [
    "QuDataInputBase",
    "QuDataOutputBase",
    "QuExecutorBase",
    "QdInBase",
    "QdOutBase",
    "QdExecBase",
]
