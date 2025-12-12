from .base_block import BaseBlock
from qudas.pipeline.artifacts import ClassicalArtifact


class ClassicalBlock(BaseBlock):
    """
    ClassicalBlock は統一部分を最小化する。

    - expected_input_type を ClassicalArtifact に固定
    - run() はユーザ定義関数 fn を呼ぶだけ
    - 古典計算（最適化 / ML / シミュレーション）を完全に自由に実装可能
    """

    expected_input_type = ClassicalArtifact

    def __init__(self, fn, **config):
        """
        Parameters
        ----------
        fn : callable
            artifact.data を入力として、古典処理を行い、結果を返す関数。
        config : dict
            任意のパラメータ。fn に渡される。
        """
        self.fn = fn
        self.config = config

    def run(self, artifact: ClassicalArtifact) -> ClassicalArtifact:
        """
        fn(artifact.data) を実行し、結果を ClassicalArtifact に包んで返す。
        """
        result = self.fn(artifact.data, **self.config)
        return ClassicalArtifact(result)