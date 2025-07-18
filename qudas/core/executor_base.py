from __future__ import annotations

from abc import ABC, abstractmethod

from .input_base import QuDataInputBase
from .output_base import QuDataOutputBase


class QuExecutorBase(ABC):
    """ゲート／アニーリング方式を問わない Executor の共通インターフェース。"""

    @abstractmethod
    def run(self, input_data: QuDataInputBase) -> QuDataOutputBase:
        """単一の入力を実行し、結果を返します。"""
        ...

    # オプショナル: 分割／並列実行 ----------------------------------------
    def run_split(
        self, input_data: QuDataInputBase, **kwargs
    ) -> QuDataOutputBase:  # noqa: D401
        """大規模入力の分割実行や並列実行を行うオプショナルメソッド。

        デフォルト実装は :class:`NotImplementedError` を送出します。
        必要な場合にサブクラスでオーバーライドしてください。
        """
        raise NotImplementedError(
            "run_split() は必要に応じてサブクラスで実装してください。"
        )


# 下位互換性維持のためのエイリアス
QdExecBase = QuExecutorBase

__all__ = ["QuExecutorBase", "QdExecBase"]