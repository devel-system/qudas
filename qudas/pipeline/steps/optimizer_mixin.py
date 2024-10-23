from abc import ABC, abstractmethod
from typing import Any
from .base import BaseStep


class OptimizerMixin(BaseStep, ABC):
    """
    パイプラインステップ内で使用する最適化処理用のMixinクラス。
    このMixinクラスは、`transform` および `optimize` メソッドをサポートし、`optimize` メソッドの実装を必須とします。
    """

    def __init__(self) -> None:
        super().__init__()
        self.models = None
        self.results = None

    def transform(self, X: Any) -> Any:
        """
        オプションの変換メソッド。サブクラスでオーバーライド可能。

        Args:
            X (Any): 入力データ。

        Returns:
            Any: 変換後のデータ。
        """
        return X  # デフォルトでは変換しない

    @abstractmethod
    def optimize(self, X: Any = None, y: Any = None, **fit_params) -> Any:
        """
        抽象的な最適化メソッド。

        Args:
            X (Any): 入力データ。
            y (Any): ターゲットデータ。
            **fit_params: 追加のパラメータ。

        Returns:
            Any: 最適化結果。
        """
        pass
