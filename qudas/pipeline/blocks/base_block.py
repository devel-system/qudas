from abc import ABC, abstractmethod


class BaseBlock(ABC):
    """
    Pipeline の Step から呼び出される最小単位のブロック。

    特徴:
    - 入力 Artifact の型を expected_input_type で宣言
    - run() で Artifact → Artifact を返す
    - Quantum / Classical の両方のブロックが継承する
    """

    expected_input_type = None  # QuantumArtifact / ClassicalArtifact を指定

    @abstractmethod
    def run(self, artifact):
        """
        artifact を入力として何らかの処理を行い、artifact を返す。

        QuantumBlock:
            QuantumArtifact → QuantumArtifact
        ClassicalBlock:
            ClassicalArtifact → ClassicalArtifact
        """
        raise NotImplementedError