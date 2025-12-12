from typing import Callable, Dict, Tuple, Type

from .artifacts import ArtifactBase, QuantumArtifact, ClassicalArtifact


class ArtifactConverterRegistry:
    """
    Artifact 間の変換を管理するレジストリ。

    例:
        QuantumArtifact -> ClassicalArtifact
        ClassicalArtifact -> QuantumArtifact（パラメータ生成など）
    """

    _registry: Dict[Tuple[Type[ArtifactBase], Type[ArtifactBase]], Callable] = {}

    @classmethod
    def register(
        cls,
        from_type: Type[ArtifactBase],
        to_type: Type[ArtifactBase],
        converter_fn: Callable[[ArtifactBase], ArtifactBase],
    ) -> None:
        """
        変換関数を登録する。

        converter_fn:
            from_type の artifact を受け取り to_type の artifact を返す関数
        """
        cls._registry[(from_type, to_type)] = converter_fn

    @classmethod
    def can_convert(
        cls,
        from_type: Type[ArtifactBase],
        to_type: Type[ArtifactBase],
    ) -> bool:
        return (from_type, to_type) in cls._registry

    @classmethod
    def convert(
        cls,
        artifact: ArtifactBase,
        to_type: Type[ArtifactBase],
    ) -> ArtifactBase:
        """
        artifact を to_type に変換する。

        - すでに to_type の場合はそのまま返す
        - 登録がなければ TypeError
        """
        if isinstance(artifact, to_type):
            return artifact

        key = (type(artifact), to_type)
        if key not in cls._registry:
            raise TypeError(
                f"No Artifact converter registered: {type(artifact).__name__} -> {to_type.__name__}"
            )

        return cls._registry[key](artifact)