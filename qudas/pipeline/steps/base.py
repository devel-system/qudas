from abc import ABC, abstractmethod


class BaseStep(ABC):
    @abstractmethod
    def set_global_params(self, params: dict) -> None:
        """
        グローバルパラメータを設定するメソッド。

        Args:
            params (dict): グローバルパラメータの辞書。
        """
        pass

    @abstractmethod
    def get_global_params(self) -> dict:
        """
        グローバルパラメータを取得するメソッド。

        Returns:
            dict: グローバルパラメータの辞書。
        """
        pass
