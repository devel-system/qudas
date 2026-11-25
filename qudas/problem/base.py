"""汎用最適化問題の基底。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class OptimizationProblem(ABC):
    """実問題の共通インターフェース。"""

    problem_id: str = "optimization"
    sense: str = "minimize"  # minimize | maximize

    @abstractmethod
    def validate(self) -> List[str]:
        """入力検証。エラーメッセージのリスト（空なら OK）。"""

    @abstractmethod
    def evaluate(self, solution: Any) -> float:
        """元問題の目的関数値を評価する。"""

    def evaluate_constraints(self, solution: Any) -> Dict[str, Any]:
        """制約評価。既定は空（制約なし）。"""
        return {"feasible": True, "violations": []}

    @abstractmethod
    def to_model(self, target: str = "qubo", **kwargs: Any) -> Any:
        """対応計算モデルへ変換（例: qubo / ising / annealing_ir）。"""

    @abstractmethod
    def decode(self, raw_solution: Any) -> Any:
        """計算結果から問題固有の解表現へ復元。"""

    def reevaluate(self, raw_solution: Any) -> Dict[str, Any]:
        """デコード + 目的関数 + 制約。"""
        decoded = self.decode(raw_solution)
        obj = self.evaluate(decoded)
        cons = self.evaluate_constraints(decoded)
        return {
            "solution": decoded,
            "objective": obj,
            "feasible": cons.get("feasible", True),
            "constraints": cons,
        }

    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "description": self.problem_id}

    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "solution": {},
                "objective": {"type": "number"},
                "feasible": {"type": "boolean"},
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"problem_id": self.problem_id, "sense": self.sense}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationProblem":
        raise NotImplementedError
