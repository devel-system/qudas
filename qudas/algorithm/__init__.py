"""Algorithm Library / Package。"""

from .base import AlgorithmSpec
from .package import AlgorithmPackage
from .registry import list_algorithms, get_algorithm, ALGORITHM_REGISTRY
from . import solvers as _solvers  # noqa: F401 — 登録副作用

__all__ = [
    "AlgorithmSpec",
    "AlgorithmPackage",
    "list_algorithms",
    "get_algorithm",
    "ALGORITHM_REGISTRY",
]
