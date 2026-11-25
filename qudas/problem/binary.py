"""Binary / Constrained Optimization の共通モデル。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import OptimizationProblem


class BinaryOptimizationProblem(OptimizationProblem):
    """バイナリ二次目的 + 任意の線形制約を持つ問題の共通表現。

    QUBO は最小化エネルギーとして扱う。
    元問題が最大化の場合は to_qubo で符号反転し、evaluate は元の sense で返す。
    """

    problem_id = "binary_optimization"
    sense = "minimize"

    def __init__(
        self,
        variables: Sequence[str],
        qubo: Optional[Dict[Tuple[str, str], float]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sense: str = "minimize",
        constant: float = 0.0,
    ) -> None:
        self.variables = list(variables)
        self.qubo: Dict[Tuple[str, str], float] = dict(qubo or {})
        self.constraints = list(constraints or [])
        self.metadata = dict(metadata or {})
        self.sense = sense
        self.constant = float(constant)

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.variables:
            errors.append("variables must be non-empty")
        var_set = set(self.variables)
        for (i, j), _ in self.qubo.items():
            if i not in var_set or j not in var_set:
                errors.append(f"QUBO key uses unknown variable: {(i, j)}")
        return errors

    def evaluate_energy(self, assignment: Dict[str, int]) -> float:
        """最小化 QUBO エネルギー（定数項込み）。"""
        energy = self.constant
        for (i, j), coeff in self.qubo.items():
            energy += coeff * int(assignment.get(i, 0)) * int(assignment.get(j, 0))
        return float(energy)

    def evaluate(self, solution: Any) -> float:
        if not isinstance(solution, dict):
            raise TypeError("solution must be dict[var, 0|1]")
        energy = self.evaluate_energy(solution)
        if self.sense == "maximize":
            return -energy
        return energy

    def evaluate_constraints(self, solution: Any) -> Dict[str, Any]:
        violations = []
        for c in self.constraints:
            kind = c.get("kind")
            if kind == "linear_le":
                # sum w_i x_i <= b
                total = sum(
                    float(w) * int(solution.get(v, 0))
                    for v, w in (c.get("weights") or {}).items()
                )
                bound = float(c.get("bound", 0))
                if total > bound + 1e-9:
                    violations.append({"constraint": c.get("name", kind), "value": total, "bound": bound})
            elif kind == "linear_eq":
                total = sum(
                    float(w) * int(solution.get(v, 0))
                    for v, w in (c.get("weights") or {}).items()
                )
                bound = float(c.get("bound", 0))
                if abs(total - bound) > 1e-9:
                    violations.append({"constraint": c.get("name", kind), "value": total, "bound": bound})
        return {"feasible": len(violations) == 0, "violations": violations}

    def to_model(self, target: str = "qubo", **kwargs: Any) -> Any:
        target = target.lower()
        if target == "qubo":
            return dict(self.qubo)
        if target in ("annealing_ir", "qudata"):
            from qudas.annealing import QdAnnealingIR

            return QdAnnealingIR(qubo=dict(self.qubo))
        if target == "ising":
            return self.to_ising()
        if target == "dimod_bqm":
            from qudas.annealing import QdAnnealingIR

            return QdAnnealingIR(qubo=dict(self.qubo)).to_dimod_bqm()
        raise ValueError(f"Unsupported target for BinaryOptimizationProblem: {target}")

    def to_ising(self) -> Dict[str, Any]:
        """dimod BQM 経由で Ising (SPIN) へ。"""
        bqm = self.to_model("dimod_bqm")
        ising = bqm.change_vartype("SPIN", inplace=False)
        linear = dict(ising.linear)
        quadratic = {tuple(k): float(v) for k, v in ising.quadratic.items()}
        return {
            "linear": linear,
            "quadratic": quadratic,
            "offset": float(ising.offset),
            "vartype": "SPIN",
        }

    def decode(self, raw_solution: Any) -> Dict[str, int]:
        if isinstance(raw_solution, dict):
            out = {}
            for v in self.variables:
                # キーが x0 / 0 / '0' など揺れうる
                if v in raw_solution:
                    out[v] = int(raw_solution[v])
                else:
                    out[v] = 0
            return out
        raise TypeError(f"Cannot decode solution of type {type(raw_solution)}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "sense": self.sense,
            "variables": self.variables,
            "qubo": [{"key": list(k), "value": v} for k, v in self.qubo.items()],
            "constant": self.constant,
            "constraints": self.constraints,
            "metadata": self.metadata,
        }
