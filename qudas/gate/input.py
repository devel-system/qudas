from qudas.core.base import QdInBase


class QuDataGateInput(QdInBase):
    def __init__(self, blocks):
        self.blocks = blocks

    def to_dict(self):
        return {"blocks": self.blocks}

    # --------------------------------------------------------------
    # ゲート用ユーティリティ (IR 変換)
    # --------------------------------------------------------------
    def to_ir(self):  # noqa: D401 – simple method name
        """保持しているブロック集合を :class:`QuAlgorithmIR` へ変換。

        各ブロックが持つ ``gates`` をフラット化し、アルゴリズム全体の IR を生成します。
        """

        from .ir import QuAlgorithmIR

        try:
            return QuAlgorithmIR.from_blocks(self.blocks)
        except Exception:
            # ブロック形式が想定外の場合は空 IR を返す – 旧実装との互換維持
            return QuAlgorithmIR(gates=[])

    @classmethod
    def from_dict(cls, data):
        return cls(blocks=data["blocks"])


QdGateIn = QuDataGateInput