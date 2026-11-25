"""TSP（巡回セールスマン問題）— Binary one-hot 定式化。

都市数 n に対し変数 x_{i,t} ∈ {0,1}: 時刻 t に都市 i を訪れる。
制約:
  - 各時刻ちょうど 1 都市
  - 各都市ちょうど 1 回訪問
目的（最小化）:
  Σ_{i,j,t} d(i,j) x_{i,t} x_{j,t+1}  （t+1 は mod n）

小規模（n<=8 程度の brute force）向け。Amplify 問題数競争はしない。
"""

from __future__ import annotations

from itertools import permutations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .binary import BinaryOptimizationProblem
from .registry import register_problem


def _var(i: int, t: int) -> str:
    return f"x_{i}_{t}"


@register_problem("tsp")
class TSPProblem(BinaryOptimizationProblem):
    problem_id = "tsp"
    sense = "minimize"

    def __init__(
        self,
        distances: Sequence[Sequence[float]],
        penalty: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        n = len(distances)
        if n < 2:
            raise ValueError("TSP requires at least 2 cities")
        for row in distances:
            if len(row) != n:
                raise ValueError("distances must be a square matrix")
        self.n = n
        self.distances = [[float(distances[i][j]) for j in range(n)] for i in range(n)]
        max_d = max(self.distances[i][j] for i in range(n) for j in range(n) if i != j)
        self.penalty = float(penalty if penalty is not None else (max_d * n + 1.0) * 10)
        variables, qubo, constraints = self._build()
        super().__init__(
            variables=variables,
            qubo=qubo,
            constraints=constraints,
            metadata=metadata or {},
            sense="minimize",
            constant=0.0,
        )

    def _build(self) -> Tuple[List[str], Dict[Tuple[str, str], float], List[Dict[str, Any]]]:
        n = self.n
        variables = [_var(i, t) for i in range(n) for t in range(n)]
        qubo: Dict[Tuple[str, str], float] = {}
        constraints: List[Dict[str, Any]] = []
        P = self.penalty

        def add(a: str, b: str, coeff: float) -> None:
            if a == b:
                key = (a, a)
            else:
                key = (a, b) if a <= b else (b, a)
            qubo[key] = qubo.get(key, 0.0) + coeff

        # 目的: 巡回距離
        for t in range(n):
            t2 = (t + 1) % n
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    add(_var(i, t), _var(j, t2), self.distances[i][j])

        def add_eq_penalty(vars_: List[str], bound: float = 1.0) -> None:
            # P (Σ x - bound)^2 （binary: x^2=x）
            for i, vi in enumerate(vars_):
                add(vi, vi, P * (1.0 - 2.0 * bound))
                for vj in vars_[i + 1 :]:
                    add(vi, vj, 2.0 * P)

        for t in range(n):
            weights = {_var(i, t): 1.0 for i in range(n)}
            constraints.append(
                {
                    "kind": "linear_eq",
                    "name": f"one_city_time_{t}",
                    "weights": weights,
                    "bound": 1.0,
                }
            )
            add_eq_penalty([_var(i, t) for i in range(n)], 1.0)

        for i in range(n):
            weights = {_var(i, t): 1.0 for t in range(n)}
            constraints.append(
                {
                    "kind": "linear_eq",
                    "name": f"visit_once_city_{i}",
                    "weights": weights,
                    "bound": 1.0,
                }
            )
            add_eq_penalty([_var(i, t) for t in range(n)], 1.0)

        return variables, qubo, constraints

    def tour_length(self, tour: Sequence[int]) -> float:
        if len(tour) != self.n:
            raise ValueError("tour length must equal number of cities")
        total = 0.0
        for t in range(self.n):
            i, j = int(tour[t]), int(tour[(t + 1) % self.n])
            total += self.distances[i][j]
        return float(total)

    def decode(self, raw_solution: Any) -> Dict[str, Any]:
        assignment = super().decode(raw_solution)
        tour = []
        for t in range(self.n):
            cities = [i for i in range(self.n) if int(assignment.get(_var(i, t), 0)) == 1]
            tour.append(cities[0] if len(cities) == 1 else None)
        return {"assignment": assignment, "tour": tour}

    def evaluate(self, solution: Any) -> float:
        if isinstance(solution, dict) and "tour" in solution:
            tour = solution["tour"]
            if None in tour:
                return float("inf")
            return self.tour_length(tour)
        if isinstance(solution, (list, tuple)):
            return self.tour_length(solution)
        decoded = self.decode(solution) if isinstance(solution, dict) else solution
        return self.evaluate(decoded)

    def evaluate_constraints(self, solution: Any) -> Dict[str, Any]:
        if isinstance(solution, dict) and "assignment" in solution:
            assignment = solution["assignment"]
        elif isinstance(solution, dict) and "tour" in solution:
            # tour → assignment
            assignment = {v: 0 for v in self.variables}
            tour = solution["tour"]
            if None not in tour and len(tour) == self.n:
                for t, city in enumerate(tour):
                    assignment[_var(int(city), t)] = 1
            else:
                return {"feasible": False, "violations": [{"constraint": "invalid_tour"}]}
        else:
            assignment = super().decode(solution) if isinstance(solution, dict) else {}
        return super().evaluate_constraints(assignment)

    def classical_brute_force(self) -> Dict[str, Any]:
        """(n-1)!/2 ではなく (n-1)! の出発固定全探索。n>9 は拒否。"""
        if self.n > 9:
            raise ValueError("classical_brute_force supports at most 9 cities")
        cities = list(range(1, self.n))
        best = float("inf")
        best_tour = None
        for perm in permutations(cities):
            tour = [0] + list(perm)
            length = self.tour_length(tour)
            if length < best:
                best = length
                best_tour = tour
        assignment = {v: 0 for v in self.variables}
        assert best_tour is not None
        for t, city in enumerate(best_tour):
            assignment[_var(city, t)] = 1
        return {
            "solution": {"assignment": assignment, "tour": best_tour},
            "objective": best,
            "feasible": True,
        }

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({"distances": self.distances, "penalty": self.penalty, "n": self.n})
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TSPProblem":
        return cls(
            distances=data["distances"],
            penalty=data.get("penalty"),
            metadata=data.get("metadata"),
        )

    @classmethod
    def from_cities(cls, coords: Sequence[Tuple[float, float]], **kwargs: Any) -> "TSPProblem":
        """ユークリッド距離行列を構築。"""
        import math

        n = len(coords)
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dist[i][j] = math.hypot(dx, dy)
        return cls(distances=dist, metadata={**(kwargs.pop("metadata", {}) or {}), "coords": list(coords)}, **kwargs)
