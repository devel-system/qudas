from .qudata_input import QuDataInput
from .qudata_output import QuDataOutput
from typing import Optional, Dict, Any

class QuData:

    @classmethod
    def input(cls, prob: Optional[Dict[str, Any]] = None) -> QuDataInput:
        """
        クラスメソッドとして QuDataInput のインスタンスを作成し、引数を受け取る。

        Args:
            prob (dict, optional): QuDataInput の引数となる最適化問題データ。

        Returns:
            QuDataInput のインスタンス。
        """
        return QuDataInput(prob)

    @classmethod
    def output(cls, result: Optional[Dict[str, Any]] = None, result_type: Optional[str] = None) -> QuDataOutput:
        """
        クラスメソッドとして QuDataOutput のインスタンスを作成し、引数を受け取る。

        Args:
            result (dict, optional): QuDataOutput の引数となる計算結果データ。
            result_type (str, optional): 結果の形式。

        Returns:
            QuDataOutput のインスタンス。
        """
        return QuDataOutput(result, result_type)