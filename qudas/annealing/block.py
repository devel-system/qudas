from typing import Union
from qudas.annealing.ir import QdAnnealingIR


class QdAnnealingBlock:
    """量子アニーリング用のブロッククラス。"""

    def __init__(self, qubo: Union[dict, QdAnnealingIR], label: str = "block"):
        """量子アニーリング用のブロッククラス。

        Parameters
        ----------
        qubo : Union[dict, QdAnnealingIR]
            QUBO を表す辞書または QdAnnealingIR オブジェクト。
        label : str, optional
            ブロックのラベル。

        Raises:
            TypeError: qubo がサポートされていない型の場合。
        """
        if isinstance(qubo, dict):
            self.qubo = QdAnnealingIR(qubo)
        elif isinstance(qubo, QdAnnealingIR):
            self.qubo = qubo
        else:
            raise TypeError(f"{type(qubo)} はサポートされていません。")
        self.label = label


QdAnnBlock = QdAnnealingBlock
