"""Algorithm Adapt: 構造を保ったまま別問題へ応用。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from qudas.algorithm.package import AlgorithmPackage
from qudas.algorithm.registry import get_algorithm
from qudas.result.common import QdExperimentResult


def adapt_qaoa_to_problem(
    problem: Any,
    *,
    parameters: Optional[Dict[str, Any]] = None,
    base_algorithm_id: str = "qaoa.qiskit",
) -> QdExperimentResult:
    """QAOA サンプル実装を別の MaxCut インスタンスへ適用する。

    回路の単純再利用ではなく、問題 Hamiltonian（辺集合）に応じて
    cost layer を再構成する（Adapt）。
    """
    if getattr(problem, "problem_id", None) != "maxcut":
        raise ValueError(
            "adapt_qaoa_to_problem currently supports MaxCutProblem only "
            f"(got {getattr(problem, 'problem_id', type(problem))})"
        )
    spec = get_algorithm(base_algorithm_id)
    pkg = AlgorithmPackage.from_spec(spec)
    params = dict(spec.parameters)
    if parameters:
        params.update(parameters)
    return pkg.run(problem, **params)
