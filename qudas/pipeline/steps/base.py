from abc import ABC


class BaseStep(ABC):
    def __init__(self):
        self.global_params = {}
        self.local_params = {}

    def set_global_params(self, params: dict) -> None:
        """
        グローバルパラメータを設定するメソッド。
        デフォルトでは何もしないため、必要なステップだけがオーバーライドする。

        Args:
            params (dict): グローバルパラメータの辞書。
        """
        self.global_params = params

    def get_global_params(self) -> dict:
        """
        グローバルパラメータを取得するメソッド。
        デフォルトでは空の辞書を返すため、必要なステップだけがオーバーライドする。

        Returns:
            dict: グローバルパラメータの辞書。
        """
        return self.global_params
