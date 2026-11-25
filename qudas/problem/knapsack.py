"""0-1 Knapsack（制約付き）。共通 BinaryOptimizationProblem を再利用。

目的: maximize Σ v_i x_i
制約: Σ w_i x_i <= capacity

QUBO へはペナルティ法で埋め込む:
  H = -Σ v_i x_i + P (Σ w_i x_i - capacity のスラック付き二乗)
簡易版ではスラック無し二乗ペナルティ（近似）ではなく、
容量超過を罰する標準的なスラック二値展開を用いる。

本実装は小規模向け: スラック変数を binary で容量まで表現。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .binary import BinaryOptimizationProblem
from .registry import register_problem


@register_problem("knapsack")
class KnapsackProblem(BinaryOptimizationProblem):
    problem_id = "knapsack"
    sense = "maximize"

    def __init__(
        self,
        values: Sequence[float],
        weights: Sequence[float],
        capacity: float,
        penalty: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(values) != len(weights):
            raise ValueError("values and weights must have same length")
        self.values = [float(v) for v in values]
        self.weights = [float(w) for w in weights]
        self.capacity = float(capacity)
        self.n = len(values)
        # ペナルティは価値スケールより十分大きく
        self.penalty = float(
            penalty if penalty is not None else (sum(abs(v) for v in self.values) + 1.0) * 10
        )
        variables, qubo, constraints = self._build()
        super().__init__(
            variables=variables,
            qubo=qubo,
            constraints=constraints,
            metadata=metadata or {},
            sense="maximize",
            constant=0.0,
        )

    def _build(self) -> Tuple[List[str], Dict[Tuple[str, str], float], List[Dict[str, Any]]]:
        item_vars = [f"x{i}" for i in range(self.n)]
        # スラック: capacity を表すのに必要なビット
        cap = int(self.capacity)
        if cap < 0:
            raise ValueError("capacity must be >= 0")
        slack_bits = max(1, cap.bit_length())
        slack_vars = [f"s{k}" for k in range(slack_bits)]
        variables = item_vars + slack_vars

        # 制約: Σ w_i x_i + Σ 2^k s_k = capacity
        # H_pen = P (Σ w_i x_i + Σ 2^k s_k - C)^2
        # 目的: H_obj = -Σ v_i x_i
        # 最小化 H = H_obj + H_pen
        qubo: Dict[Tuple[str, str], float] = {}

        def add(a: str, b: str, coeff: float) -> None:
            if a == b:
                key = (a, a)
            else:
                key = (a, b) if a <= b else (b, a)
            qubo[key] = qubo.get(key, 0.0) + coeff

        # 線形目的（最大化 → 最小化で符号反転）
        for i, v in enumerate(self.values):
            add(item_vars[i], item_vars[i], -v)

        # 線形和 L = Σ w_i x_i + Σ 2^k s_k
        linear_terms: List[Tuple[str, float]] = []
        for i, w in enumerate(self.weights):
            linear_terms.append((item_vars[i], w))
        for k, s in enumerate(slack_vars):
            linear_terms.append((s, float(1 << k)))

        P = self.penalty
        C = self.capacity
        # P (L - C)^2 = P (L^2 - 2 C L + C^2)
        # 定数 P C^2 は energy に別途載せるが、比較では相対で十分なため constant に保持
        self.constant = P * (C ** 2)

        # P (L - C)^2 = P L^2 - 2 P C L + P C^2
        # L^2 = Σ_i wa^2 xa^2 + 2 Σ_{a<b} wa wb xa xb （binary では xa^2=xa）
        for a, wa in linear_terms:
            add(a, a, -2.0 * P * C * wa)
            add(a, a, P * (wa ** 2))
        for i, (a, wa) in enumerate(linear_terms):
            for b, wb in linear_terms[i + 1 :]:
                add(a, b, 2.0 * P * wa * wb)

        constraints = [
            {
                "kind": "linear_le",
                "name": "capacity",
                "weights": {item_vars[i]: self.weights[i] for i in range(self.n)},
                "bound": self.capacity,
            }
        ]
        return variables, qubo, constraints

    def decode(self, raw_solution: Any) -> Dict[str, Any]:
        bits = super().decode(raw_solution)
        selected = [i for i in range(self.n) if bits.get(f"x{i}", 0) == 1]
        return {
            "assignment": {f"x{i}": bits.get(f"x{i}", 0) for i in range(self.n)},
            "selected_items": selected,
            "total_value": sum(self.values[i] for i in selected),
            "total_weight": sum(self.weights[i] for i in selected),
        }

    def evaluate(self, solution: Any) -> float:
        if isinstance(solution, dict) and "total_value" in solution:
            return float(solution["total_value"])
        if isinstance(solution, dict) and "assignment" in solution:
            sol = solution["assignment"]
        else:
            sol = solution
        return float(
            sum(self.values[i] * int(sol.get(f"x{i}", 0)) for i in range(self.n))
        )

    def evaluate_constraints(self, solution: Any) -> Dict[str, Any]:
        if isinstance(solution, dict) and "assignment" in solution:
            sol = solution["assignment"]
        elif isinstance(solution, dict) and "selected_items" in solution:
            sol = {f"x{i}": 1 if i in solution["selected_items"] else 0 for i in range(self.n)}
        else:
            sol = solution
        return super().evaluate_constraints(sol)

    def classical_brute_force(self) -> Dict[str, Any]:
        if self.n > 20:
            raise ValueError("classical_brute_force supports at most 20 items")
        best_val = float("-inf")
        best: Dict[str, Any] = {}
        for mask in range(1 << self.n):
            selected = [i for i in range(self.n) if (mask >> i) & 1]
            weight = sum(self.weights[i] for i in selected)
            if weight > self.capacity + 1e-9:
                continue
            val = sum(self.values[i] for i in selected)
            if val > best_val:
                best_val = val
                assignment = {f"x{i}": (mask >> i) & 1 for i in range(self.n)}
                best = {
                    "solution": {
                        "assignment": assignment,
                        "selected_items": selected,
                        "total_value": val,
                        "total_weight": weight,
                    },
                    "objective": val,
                    "feasible": True,
                }
        return best

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "sense": self.sense,
            "values": self.values,
            "weights": self.weights,
            "capacity": self.capacity,
            "penalty": self.penalty,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnapsackProblem":
        return cls(
            values=data["values"],
            weights=data["weights"],
            capacity=data["capacity"],
            penalty=data.get("penalty"),
            metadata=data.get("metadata"),
        )
