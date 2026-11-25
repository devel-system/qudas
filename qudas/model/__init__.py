"""交換用 Model 仕様（ONNX 型の Envelope / 変換レポート）。"""

from .schema import SCHEMA_VERSION, ModelType, ConversionKind
from .envelope import QdModelEnvelope
from .conversion import ConversionReport, ConversionError

__all__ = [
    "SCHEMA_VERSION",
    "ModelType",
    "ConversionKind",
    "QdModelEnvelope",
    "ConversionReport",
    "ConversionError",
]
