from abc import ABC, abstractmethod


class BaseStep(ABC):
    @abstractmethod
    def set_global_params(self, params):
        """グローバルパラメータを設定するメソッド"""
        pass

    @abstractmethod
    def get_global_params(self):
        """グローバルパラメータを取得するメソッド"""
        pass
