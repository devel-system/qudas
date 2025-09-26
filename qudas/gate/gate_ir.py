from typing import List, Optional


class QdGateIR:
    """単一ゲートのIR表現。"""

    def __init__(
        self,
        gate: str,
        targets: List[int],
        controls: Optional[List[int]] = None,
        params: Optional[List[float]] = None,
    ) -> None:
        """QdGateIR を初期化する。

        Parameters
        ----------
        gate : str
            ゲート名 (例: "cx", "ry" など)
        targets : List[int]
            対象となる量子ビットのインデックス (0 始まり)
        controls : Optional[List[int]], default None
            制御ビットのインデックス (制御ゲートの場合)。指定がなければ空リスト。
        params : Optional[List[float]], default None
            ゲートに付随するパラメータ (回転角など)。指定がなければ空リスト。
        """
        self.gate: str = gate
        self.targets: List[int] = targets
        self.controls: List[int] = controls or []
        self.params: List[float] = params or []
