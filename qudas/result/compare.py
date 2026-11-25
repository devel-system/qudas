"""複数 Result の比較。"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .common import QdExperimentResult


def compare_results(
    results: Sequence[QdExperimentResult],
    *,
    key: str = "problem_objective",
) -> Dict[str, Any]:
    """同一問題上の Result を比較用テーブルにまとめる。

    Parameters
    ----------
    results :
        比較対象。
    key :
        並び替えキー（既定は問題目的関数値）。
    """
    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "algorithm": r.algorithm,
                "backend": r.backend,
                "status": r.status,
                "objective_value": r.objective_value,
                "problem_objective": r.problem_objective,
                "minimized_energy": r.minimized_energy,
                "feasibility": r.feasibility,
                "sense": r.sense,
                "timing_total_s": r.timing.total_s,
                "resources": r.resources.to_dict(),
                "parameters": r.parameters,
                "random_seed": r.random_seed,
            }
        )

    def sort_key(row: Dict[str, Any]):
        v = row.get(key)
        return (v is None, v if v is not None else 0)

    # maximize なら降順、混在時は problem_objective の大きい方を先に
    sense = results[0].sense if results else "maximize"
    reverse = sense == "maximize"
    ranked = sorted(rows, key=sort_key, reverse=reverse)

    return {
        "count": len(rows),
        "sort_key": key,
        "sense": sense,
        "rows": ranked,
        "best": ranked[0] if ranked else None,
    }
