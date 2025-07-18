from __future__ import annotations

from typing import List, Dict, Union, Optional

from qudas.core.base import QdInBase
from qudas.annealing.ir import QdAnnIR
from qudas.annealing.block import QdAnnBlock


class QuDataAnnealingInput(QdInBase):
    """量子アニーリング (QUBO) 用の入力クラス。

    旧 API では単一 QUBO (``QdAnnIR``) のみを扱っていたが、
    本クラスでは *複数ブロック* を ``QdAnnBlock`` のリストとして
    扱えるように拡張した。

    Parameters
    ----------
    blocks : list[QdAnnBlock] | QdAnnIR | None, optional
        ・``list`` を渡した場合        … 複数ブロック入力としてそのまま保持。
        ・``QdAnnIR`` を渡した場合  … 旧 API 互換。単一ブロックとしてラップ。
        ・省略 / None                 … 空ブロックリストで初期化。
    """

    def __init__(self, blocks: Union[List[QdAnnBlock], QdAnnIR, None] = None):
        if blocks is None:
            self.blocks: List[QdAnnBlock] = []
        # 新 API: list[QdAnnBlock]
        elif isinstance(blocks, list):
            if not all(isinstance(b, QdAnnBlock) for b in blocks):
                raise TypeError("blocks には QdAnnBlock のリストを渡してください。")
            self.blocks = blocks
        # 旧 API: 単一 QdAnnIR
        elif isinstance(blocks, QdAnnIR):
            self.blocks = [QdAnnBlock(blocks.to_dict(), label="block0")]
        else:
            raise TypeError(
                "blocks には QdAnnBlock のリスト、QdAnnIR、または None を渡してください。"
            )

    # ------------------------------------------------------------------
    # 旧 API 互換: `.ir` プロパティ (最初のブロックを参照)
    # ------------------------------------------------------------------
    @property
    def ir(self) -> Optional[QdAnnIR]:  # noqa: D401 – simple property
        """互換用プロパティ: **最初のブロック** を ``QdAnnIR`` として返す。"""
        if not self.blocks:
            return None
        return QdAnnIR(self.blocks[0].qubo)

    # ------------------------------------------------------------------
    # 汎用ユーティリティ
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Dict]:  # noqa: D401 – 単純メソッド
        """``{block_label: qubo_dict}`` 形式へ変換。"""
        return {block.label: block.qubo for block in self.blocks}


# エイリアス (旧クラス名)
QdAnnIn = QuDataAnnealingInput