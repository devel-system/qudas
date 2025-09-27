from qudas.core.base import QdInputBase
from qudas.gate.block import QdGateBlock
from qudas.gate.ir import QdAlgorithmIR
from typing import List, Union


class QdGateInput(QdInputBase):
    """量子ゲート方式の入力クラス。"""

    def __init__(self, blocks: Union[List[QdGateBlock], QdAlgorithmIR, None] = None):
        """量子ゲート方式の入力クラス。

        Parameters
        ----------
        blocks : list[QdGateBlock] | QdAlgorithmIR | None
            量子ゲートブロックのリスト。

        Raises:
            TypeError: blocks に QdGateBlock のリスト以外を渡した場合。
            TypeError: blocks に QdAlgorithmIR 以外を渡した場合。
        """
        if blocks is None:
            self.blocks: List[QdGateBlock] = []
        elif isinstance(blocks, list):
            if not all(isinstance(b, QdGateBlock) for b in blocks):
                raise TypeError("blocks には QdGateBlock のリストを渡してください。")
            self.blocks = blocks
        elif isinstance(blocks, QdAlgorithmIR):
            self.blocks = [QdGateBlock(gates=blocks.gates)]
        else:
            raise TypeError(
                "blocks には QdGateBlock のリスト、QdAlgorithmIR、または None を渡してください。"
            )

    @property
    def block(self) -> QdGateBlock:
        return self.blocks[0]

    def to_dict(self):
        return {
            block.label: {"gates": block.gates, "num_qubits": block.num_qubits}
            for block in self.blocks
        }

    # --------------------------------------------------------------
    # ゲート用ユーティリティ (IR 変換)
    # --------------------------------------------------------------
    def to_ir(self):  # noqa: D401 – simple method name
        """保持しているブロック集合を :class:`QdAlgorithmIR` へ変換。

        各ブロックが持つ ``gates`` をフラット化し、アルゴリズム全体の IR を生成します。
        """

        from .ir import QdAlgorithmIR

        try:
            return QdAlgorithmIR.from_blocks(self.blocks)
        except Exception:
            # ブロック形式が想定外の場合は空 IR を返す – 旧実装との互換維持
            return QdAlgorithmIR(gates=[])

    @classmethod
    def from_dict(cls, data):
        return cls(blocks=data["blocks"])


QdGateIn = QdGateInput
