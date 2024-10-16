from abc import ABC, abstractmethod
from typing import Dict, Any


class QuDataBase(ABC):
    def __init__(self, data: Dict[str, Any] = None):
        """
        初期データを格納する抽象クラス。

        Args:
            data (dict, optional): 入力または出力データ。デフォルトはNone。
        """
        self.data = {}

        if data is None:
            self.data = {}
        elif isinstance(data, dict):
            self.data = data
        else:
            raise TypeError(f"{type(data)}は対応していない型です。")

    # @abstractmethod
    # def from_xxx(self, data):
    #     """サブクラスで実装されるべき抽象メソッド"""
    #     pass

    # @abstractmethod
    # def to_xxx(self):
    #     """サブクラスで実装されるべき抽象メソッド"""
    #     pass
