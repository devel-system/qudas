from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class QdInputBase(ABC):
    """qudas の各 Executor に入力されるデータの基底クラス。

    ゲート方式、アニーリングのQUBO方式のどちらでも利用できる共通インターフェースを提供します。
    ゲート系を対象とするサブクラスでは ``to_ir`` をオーバーライドして、
    ゲート用の中間表現 (IR) を返すように実装してください。
    """

    @abstractmethod
    def to_dict(self) -> dict:
        """オブジェクトを辞書にシリアライズします。"""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "QdInputBase":
        """``to_dict`` で得られた辞書からインスタンスを復元します。"""
        ...

    def to_ir(self) -> Any:
        """入力をゲート回路などの中間表現に変換します。

        デフォルト実装は :class:`NotImplementedError` を送出します。
        ゲート系を扱うサブクラスで必要に応じてオーバーライドしてください。
        """
        raise NotImplementedError(
            "to_ir() はゲート方式のサブクラスでオーバーライドしてください。"
        )


# 下位互換性維持のためのエイリアス
QdInBase = QdInputBase

__all__ = ["QdInputBase", "QdInBase"]
