from qudas.annealing import QdAnnIR, QdAnnealingOutput
from typing import Optional, Dict, Any


class QuData:

    @classmethod
    def input(cls, prob: Optional[Dict[str, Any]] = None) -> QdAnnIR:
        """
        新IR (QdAnnIR) を返却するラッパー。旧API互換のために残してある。
        """
        if prob is None:
            return QdAnnIR()
        if isinstance(prob, dict):
            return QdAnnIR(prob)
        raise TypeError(f"{type(prob)}は対応していない型です。")

    @classmethod
    def output(
        cls,
        # 新 API
        results: Optional[Dict[str, Dict[str, Any]]] = None,
        # 旧 API
        result: Optional[Dict[str, Any]] = None,
        result_type: Optional[str] = None,
        **kwargs,
    ) -> QdAnnealingOutput:
        """新しい出力クラス (QuDataAnnealingOutput) を返却する。

        旧 API の `result`/`result_type` でも呼び出せるように互換を維持する。
        """

        return QdAnnealingOutput(
            results=results,
            result=result,
            result_type=result_type,
            **kwargs,
        )

# 旧クラス名のエイリアス（互換性維持）
QuDataInput = QdAnnIR
QuDataOutput = QdAnnealingOutput
