from typing import Any, Dict, Optional


class ArtifactBase:
    """
    Pipeline 内で受け渡しされる最小単位のデータコンテナ。

    - data: 実データ（量子 Output / numpy array / dict 等）
    - metadata: 任意の補助情報（backend, shot数, seed など）
    """

    def __init__(self, data: Any, metadata: Optional[Dict[str, Any]] = None):
        self.data = data
        self.metadata: Dict[str, Any] = metadata or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={type(self.data)}, metadata={list(self.metadata.keys())})"


class QuantumArtifact(ArtifactBase):
    """
    量子計算結果用 Artifact。

    data:
        - QdGateOutput
        - QdAnnealingOutput
        - もしくはそれらに準ずる構造
    """
    pass


class ClassicalArtifact(ArtifactBase):
    """
    古典計算結果用 Artifact。

    data:
        - scalar / list / dict / numpy.ndarray
        - ユーザ定義の任意オブジェクト
    """
    pass