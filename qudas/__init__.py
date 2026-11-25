# __init__.py

from __future__ import annotations

import os


def _read_version() -> str:
    """開発ツリーの VERSION を優先し、無ければインストールメタデータを使う。"""
    version_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "VERSION"
    )
    try:
        with open(version_file_path, encoding="utf-8") as version_file:
            text = version_file.read().strip()
            if text:
                return text
    except FileNotFoundError:
        pass

    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        from importlib_metadata import PackageNotFoundError, version  # type: ignore

    try:
        return version("qudas")
    except PackageNotFoundError:
        return "0.0.0+unknown"


__version__ = _read_version()

from .gate import (
    QdGateInput,
    QdGateOutput,
    QdGateExecutor,
    QdAlgorithmIR,
    QdGateBlock,
    QdGateIR,
)
from .annealing import (
    QdAnnealingInput,
    QdAnnealingOutput,
    QdAnnealingExecutor,
    QdAnnealingIR,
    QdAnnealingBlock,
)
from .qudata import QuData
from .pipeline import QdPipeline, QdPipelineModel

# 新機能サブパッケージは import 可能にするが、トップレベルは既存 Stable を維持。
# 高レベル API は qudas.api / qudas.problem / qudas.algorithm から利用。

__all__ = [
    "QuData",
    "QdPipeline",
    "QdPipelineModel",
    "QdGateInput",
    "QdGateOutput",
    "QdGateExecutor",
    "QdAlgorithmIR",
    "QdGateBlock",
    "QdGateIR",
    "QdAnnealingInput",
    "QdAnnealingOutput",
    "QdAnnealingExecutor",
    "QdAnnealingIR",
    "QdAnnealingBlock",
    "__version__",
]
