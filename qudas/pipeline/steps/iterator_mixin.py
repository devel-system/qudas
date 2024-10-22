from abc import ABC, abstractmethod
from typing import Any, Tuple
from .base import BaseStep


class IteratorMixin(BaseStep, ABC):
    """
    パイプラインステップ内で使用するイテレータ用のMixinクラス。
    このMixinクラスは次のパラメータセットを生成する `next_params` メソッドの実装を要求します。
    """

    def __init__(self, loop_num: int = 1) -> None:
        super().__init__()
        self.loop_num = loop_num  # デフォルトで1回のループ
        self.models = None
        self.results = None

    @abstractmethod
    def next_params(self, X: Any, y: Any = None, **iter_params) -> Tuple[Any, Any]:
        """
        次のパラメータセットを生成する抽象メソッド。

        Args:
            X (Any): 入力データ。
            y (Any): ターゲットデータ。
            **iter_params: 追加のパラメータ。

        Returns:
            Tuple[Any, Any]: 次の入力データとターゲットデータ。
        """
        pass
