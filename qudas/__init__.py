# __init__.py

# バージョン情報をVERSIONファイルから読み込む
import os

# VERSIONファイルのパスを絶対パスで取得
VERSION_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'VERSION'
)

# VERSIONファイルからバージョンを読み込む
try:
    with open(VERSION_FILE_PATH) as version_file:
        __version__ = version_file.read().strip()
except FileNotFoundError:
    __version__ = '0.0.1'  # デフォルトバージョンを設定

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
from .pipeline import Pipeline

__all__ = [
    "QuData",
    "Pipeline",
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
]
