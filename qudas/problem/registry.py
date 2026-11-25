"""Problem クラスの登録。"""

from __future__ import annotations

from typing import Dict, List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import OptimizationProblem

PROBLEM_REGISTRY: Dict[str, Type["OptimizationProblem"]] = {}


def register_problem(problem_id: str):
    def deco(cls: Type["OptimizationProblem"]) -> Type["OptimizationProblem"]:
        PROBLEM_REGISTRY[problem_id] = cls
        cls.problem_id = problem_id  # type: ignore[attr-defined]
        return cls

    return deco


def get_problem_class(problem_id: str) -> Type["OptimizationProblem"]:
    if problem_id not in PROBLEM_REGISTRY:
        raise KeyError(f"Unknown problem_id: {problem_id}")
    return PROBLEM_REGISTRY[problem_id]


def list_problems() -> List[Dict[str, str]]:
    # 遅延 import でレジストリを埋める
    from . import maxcut as _maxcut  # noqa: F401
    from . import knapsack as _knapsack  # noqa: F401

    return [
        {"problem_id": pid, "class": cls.__name__}
        for pid, cls in sorted(PROBLEM_REGISTRY.items())
    ]
