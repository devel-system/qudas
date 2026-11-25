"""Model 交換スキーマの定数。"""

from __future__ import annotations

from enum import Enum

SCHEMA_VERSION = "1.0.0"


class ModelType(str, Enum):
    PIPELINE = "pipeline"
    GATE = "gate"
    ANNEALING = "annealing"
    CLASSICAL = "classical"
    PROBLEM = "problem"
    ALGORITHM_PACKAGE = "algorithm_package"
    RESULT = "result"
    ENVELOPE = "envelope"


class ConversionKind(str, Enum):
    LOSSLESS = "lossless"
    APPROXIMATE = "approximate"
    PROBLEM_LEVEL = "problem_level"
    UNSUPPORTED = "unsupported"
