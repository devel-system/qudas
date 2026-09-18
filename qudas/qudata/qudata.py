from typing import Optional, Dict, Any, Union

from qudas.annealing import QdAnnealingIR, QdAnnealingOutput
from qudas.gate import QdGateOutput


class QuData:
    """変換専用ユーザ向け Public 入口。

    実装の正は annealing 側（``QdAnnealingIR`` / ``QdAnnealingOutput``）。
    本クラスはそれらへの facade であり、Executor による実行は行わない。
    回路・実行は ``qudas.gate`` / ``qudas.annealing`` の各 API を使う。
    """

    @classmethod
    def input(
        cls, prob: Optional[Dict[str, Any]] = None, mode: str = "annealing"
    ) -> QdAnnealingIR:
        """QUBO 変換用 IR（``QdAnnealingIR``）を返す。

        Parameters
        ----------
        prob :
            初期 QUBO dict（省略可）。
        mode :
            ``\"annealing\"`` のみサポート。ゲート回路は ``qudas.gate`` を使用すること。
        """
        if mode == "annealing":
            return QdAnnealingIR(prob if prob else {})
        if mode == "gate":
            raise NotImplementedError(
                "QuData.input(mode='gate') is not supported. "
                "Use qudas.gate (QdGateIR / QdGateBlock / QdGateInput) instead."
            )
        raise ValueError(f"Unsupported QuData.input mode: {mode!r}")

    @classmethod
    def output(
        cls,
        result: Optional[Dict[str, Any]] = None,
        result_type: Optional[str] = None,
        mode: str = "annealing",
        **kwargs,
    ) -> Union[QdAnnealingOutput, QdGateOutput]:
        """出力 facade。既定は ``QdAnnealingOutput``。

        旧 API の ``result`` / ``result_type`` でも呼び出せるように互換を維持する。
        ``result_type`` は現行 AnnealingOutput では未使用（互換のため受け取る）。
        """
        del result_type  # 互換シグネチャ維持

        if mode == "annealing":
            return QdAnnealingOutput(results={"block0": result}, **kwargs)
        if mode == "gate":
            return QdGateOutput(results={"block0": result}, **kwargs)
        raise ValueError(f"Unsupported QuData.output mode: {mode!r}")
