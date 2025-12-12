# qudas/pipeline/blocks/quantum_block.py
from .base_block import BaseBlock
from qudas.pipeline.artifacts import QuantumArtifact

# 量子側の既存構造を利用
# gate / annealing のどちらにも対応できるよう import は抽象化可能
# （ユーザが明示的に executor/input を渡す設計で柔軟性確保）
class QuantumBlock(BaseBlock):
    """
    量子計算ブロック。

    既存 Qudas の「Input → IR → Executor → Output」構造をそのまま利用する。
    Pipeline は QuantumArtifact のみ受け渡し、
    QuantumBlock 内部で Executor を呼び出すだけ。
    """

    expected_input_type = QuantumArtifact

    def __init__(self, input_obj, executor_cls, ir_builder=None):
        """
        Parameters
        ----------
        input_obj : QdGateInput / QdAnnealingInput
        executor_cls : QdGateExecutor / QdAnnealingExecutor
        ir_builder : IR を構築する callable（必要に応じて）
        """
        self.input = input_obj
        self.executor = executor_cls()
        self.ir_builder = ir_builder

    def run(self, artifact: QuantumArtifact) -> QuantumArtifact:
        """
        Pipeline から呼ばれ、量子計算を 1 回実行する。
        """

        # IR 構築（必要なら）
        ir = self.ir_builder(self.input, artifact.data) if self.ir_builder else self.input

        # Executor で量子実行
        output = self.executor.run(ir)

        # 出力を QuantumArtifact で包んで返す
        return QuantumArtifact(output)