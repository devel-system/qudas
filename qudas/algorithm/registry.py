"""Algorithm 登録。"""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import AlgorithmSpec

ALGORITHM_REGISTRY: Dict[str, "AlgorithmSpec"] = {}


def register_algorithm(spec: "AlgorithmSpec") -> "AlgorithmSpec":
    ALGORITHM_REGISTRY[spec.algorithm_id] = spec
    return spec


def get_algorithm(algorithm_id: str) -> "AlgorithmSpec":
    if algorithm_id not in ALGORITHM_REGISTRY:
        raise KeyError(
            f"Unknown algorithm_id={algorithm_id!r}. "
            f"Available: {sorted(ALGORITHM_REGISTRY)}"
        )
    return ALGORITHM_REGISTRY[algorithm_id]


def list_algorithms() -> List[Dict[str, str]]:
    return [
        {
            "algorithm_id": s.algorithm_id,
            "name": s.name,
            "version": s.version,
            "compatible_problems": ",".join(s.compatible_problems),
        }
        for s in sorted(ALGORITHM_REGISTRY.values(), key=lambda x: x.algorithm_id)
    ]
