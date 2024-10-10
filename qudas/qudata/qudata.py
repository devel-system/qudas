from .qudata_input import QuDataInput
from .qudata_output import QuDataOutput
from typing import Optional, Dict, Any

class QuData:
    def __init__(self) -> None:
        """
        QuDataInput と QuDataOutput を統合するクラス。
        初期化時に、両方のインスタンスを作成します。
        """
        self._input: Optional[QuDataInput] = None
        self._output: Optional[QuDataOutput] = None

    def input(self, prob: Optional[Dict[str, Any]] = None) -> QuDataInput:
        """
        QuDataInput のインスタンスを作成し、引数を受け取る。

        Args:
            prob (dict, optional): QuDataInput の引数となる最適化問題データ。

        Returns:
            QuDataInput のインスタンス。
        """
        self._input = QuDataInput(prob)
        return self._input

    def output(self, result: Optional[Dict[str, Any]] = None, result_type: Optional[str] = None) -> QuDataOutput:
        """
        QuDataOutput のインスタンスを作成し、引数を受け取る。

        Args:
            result (dict, optional): QuDataOutput の引数となる計算結果データ。
            result_type (str, optional): 結果の形式。

        Returns:
            QuDataOutput のインスタンス。
        """
        self._output = QuDataOutput(result, result_type)
        return self._output