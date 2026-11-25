"""交換用 Model 仕様（ONNX 型の Envelope / 変換レポート）。"""

from .schema import SCHEMA_VERSION, ModelType, ConversionKind
from .envelope import QdModelEnvelope
from .conversion import ConversionReport, ConversionError
from .graph import (
    ConversionGraph,
    ConversionPath,
    ConversionEdge,
    default_conversion_graph,
    get_default_graph,
    weakest_kind,
)

__all__ = [
    "SCHEMA_VERSION",
    "ModelType",
    "ConversionKind",
    "QdModelEnvelope",
    "ConversionReport",
    "ConversionError",
    "ConversionGraph",
    "ConversionPath",
    "ConversionEdge",
    "default_conversion_graph",
    "get_default_graph",
    "weakest_kind",
]
