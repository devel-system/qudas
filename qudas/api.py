"""利用者向け高レベル API。

初心者: Problem → Algorithm → 実行 → 比較
上級者: Pipeline Model / Algorithm Package を直接操作
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from qudas.algorithm.package import AlgorithmPackage
from qudas.algorithm.registry import get_algorithm, list_algorithms
from qudas.problem.registry import list_problems
from qudas.result.common import QdExperimentResult
from qudas.result.compare import compare_results


def solve(
    problem: Any,
    algorithm: str = "classical.exact",
    **params: Any,
) -> QdExperimentResult:
    """問題を指定 Algorithm で解く。"""
    spec = get_algorithm(algorithm)
    pkg = AlgorithmPackage.from_spec(spec)
    return pkg.run(problem, **params)


def solve_many(
    problem: Any,
    algorithms: Optional[Sequence[str]] = None,
    **common_params: Any,
) -> List[QdExperimentResult]:
    """複数 Algorithm で解き、Result リストを返す。"""
    if algorithms is None:
        algorithms = ["classical.exact", "annealing.dimod_exact"]
        # qaoa は MaxCut かつ qiskit があるときのみ
        if getattr(problem, "problem_id", None) == "maxcut":
            algorithms = list(algorithms) + ["qaoa.qiskit"]
    results = []
    for alg in algorithms:
        try:
            results.append(solve(problem, alg, **common_params))
        except Exception as exc:  # noqa: BLE001
            results.append(
                QdExperimentResult(
                    status="error",
                    message=str(exc),
                    algorithm=alg,
                    sense=getattr(problem, "sense", "maximize"),
                )
            )
    return results


def compare(problem: Any, algorithms: Optional[Sequence[str]] = None, **params: Any) -> Dict[str, Any]:
    results = solve_many(problem, algorithms=algorithms, **params)
    return {"results": [r.to_dict() for r in results], "comparison": compare_results(results)}


def catalog() -> Dict[str, Any]:
    return {
        "problems": list_problems(),
        "algorithms": list_algorithms(),
    }
