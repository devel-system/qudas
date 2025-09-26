from qudas.annealing import QdAnnealingIR, QdAnnealingOutput
from qudas.gate import QdGateIR, QdGateOutput
from typing import Optional, Dict, Any, Union


class QuData:
    """gate/annealing 共通フロントエンド"""

    @classmethod
    def input(
        cls, prob: Optional[Dict[str, Any]] = None, mode: str = "annealing"
    ) -> Union[QdAnnealingIR, QdGateIR]:
        """
        新IR (QdAnnealingIR) を返却するラッパー。旧API互換のために残してある。
        """
        if mode == "annealing":
            return QdAnnealingIR(prob if prob else {})
        elif mode == "gate":
            return QdGateIR(prob if prob else {})
        else:
            raise TypeError(f"{type(prob)}は対応していない型です。")

    @classmethod
    def output(
        cls,
        result: Optional[Dict[str, Any]] = None,
        result_type: Optional[str] = None,
        mode: str = "annealing",
        **kwargs,
    ) -> Union[QdAnnealingOutput, QdGateOutput]:
        """新しい出力クラス (QuDataAnnealingOutput) を返却する。

        旧 API の `result`/`result_type` でも呼び出せるように互換を維持する。
        """

        if mode == "annealing":
            return QdAnnealingOutput(results={"block0": result}, **kwargs)
        elif mode == "gate":
            return QdGateOutput(results={"block0": result}, **kwargs)
        else:
            raise TypeError(f"{type(result)}は対応していない型です。")
