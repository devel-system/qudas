from __future__ import annotations


from qudas.core.base import QdInputBase
from qudas.annealing.block import QdAnnealingBlock
from qudas.annealing.ir import QdAnnealingIR
from typing import List, Dict, Union, Optional


class QdAnnealingInput(QdInputBase):
    """量子アニーリング (QUBO) 用の入力クラス。

    旧 API では単一 QUBO (``QdAnnealingIR``) のみを扱っていたが、
    本クラスでは *複数ブロック* を ``QdAnnealingBlock`` のリストとして
    扱えるように拡張した。

    Parameters
    ----------
    blocks : list[QdAnnealingBlock] | QdAnnealingIR | None, optional
        ・``list`` を渡した場合        … 複数ブロック入力としてそのまま保持。
        ・``QdAnnealingIR`` を渡した場合  … 旧 API 互換。単一ブロックとしてラップ。
        ・省略 / None                 … 空ブロックリストで初期化。
    """

    def __init__(
        self, blocks: Union[List[QdAnnealingBlock], QdAnnealingIR, None] = None
    ):
        if blocks is None:
            self.blocks: List[QdAnnealingBlock] = []
        # 新 API: list[QdAnnealingBlock]
        elif isinstance(blocks, list):
            if not all(isinstance(b, QdAnnealingBlock) for b in blocks):
                raise TypeError(
                    "blocks には QdAnnealingBlock のリストを渡してください。"
                )
            self.blocks = blocks
        # 旧 API: 単一 QdAnnealingIR
        elif isinstance(blocks, QdAnnealingIR):
            self.blocks = [QdAnnealingBlock(blocks.to_dict(), label="block0")]
        else:
            raise TypeError(
                "blocks には QdAnnealingBlock のリスト、QdAnnealingIR、または None を渡してください。"
            )

    @property
    def block(self) -> QdAnnealingBlock:
        return self.blocks[0]

    # ------------------------------------------------------------------
    # 旧 API 互換: `.ir` プロパティ (最初のブロックを参照)
    # ------------------------------------------------------------------
    @property
    def ir(self) -> Optional[QdAnnealingIR]:  # noqa: D401 – simple property
        """互換用プロパティ: **最初のブロック** を ``QdAnnealingIR`` として返す。"""
        if not self.blocks:
            return None
        return QdAnnealingIR(self.blocks[0].qubo)

    # ------------------------------------------------------------------
    # 汎用ユーティリティ
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Dict]:  # noqa: D401 – 単純メソッド
        """``{block_label: qubo_dict}`` 形式へ変換。"""
        return {block.label: block.qubo for block in self.blocks}

    @classmethod
    def from_dict(cls, data: Dict[str, Dict]) -> "QdAnnealingInput":
        return cls(
            blocks=[QdAnnealingBlock(qubo, label=label) for label, qubo in data.items()]
        )


# エイリアス (旧クラス名)
QdAnnIn = QdAnnealingInput
