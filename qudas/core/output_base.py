from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Dict


@dataclass
class QdOutputBaseData:
    solution: Any
    device: str


class QdOutputBase(ABC):
    """qudas Executor から返却される結果データの基底クラス。

    ゲート・アニーリング方式に依存しない共通機能として
    ""``to_dict`` での辞書変換`` と ``visualize`` による可視化を提供します。"""

    @abstractmethod
    def to_dict(self) -> dict:
        """結果を辞書へシリアライズします。"""
        ...

    @abstractmethod
    def to_sdk_format(self, target: str):
        """外部 SDK 向けのフォーマットに変換します。

        例: ``target="qiskit"`` や ``target="amplify"`` など
        """
        ...

    @classmethod
    @abstractmethod
    def from_sdk_format(cls, sdk_obj: Any, target: str) -> "QdOutputBase":
        """外部 SDK 向けのフォーマットからインスタンスを生成します。"""
        ...

    @abstractmethod
    def visualize(self):
        """結果を可視化します。"""
        ...


# 下位互換性維持のためのエイリアス
QdOutBase = QdOutputBase

__all__ = ["QdOutputBase", "QdOutBase"]
