"""シリアライズ可能な組み込み Step。

実行時は context['problem'] から問題インスタンスを受け取る。
任意のユーザー定義 Python オブジェクトは保存しない。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from qudas.pipeline.base import QdBaseStep, OptimizerStep
from qudas.pipeline.registry import register_builtin
from qudas.result.common import QdExperimentResult


def _problem_from_context(step: QdBaseStep):
    ctx = step.get_context() if hasattr(step, "get_context") else {}
    problem = None
    if isinstance(ctx, dict):
        problem = ctx.get("problem")
        params = ctx.get("params") or {}
        if problem is None:
            problem = params.get("problem")
    if problem is None and hasattr(step, "global_params"):
        problem = (step.global_params or {}).get("problem")
    if problem is None:
        raise RuntimeError(
            "Built-in algorithm steps require context['problem'] "
            "(or params['problem']) to be set before run"
        )
    return problem


@register_builtin("qudas.algorithm.classical_exact")
class ClassicalExactStep(OptimizerStep):
    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = dict(params)

    def optimize(self, X=None, y=None):
        from qudas.algorithm.runners import run_classical_exact

        problem = X if hasattr(X, "classical_brute_force") else _problem_from_context(self)
        return run_classical_exact(problem, self.params)


@register_builtin("qudas.algorithm.annealing_dimod")
class AnnealingDimodStep(OptimizerStep):
    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = dict(params)

    def optimize(self, X=None, y=None):
        from qudas.algorithm.runners import run_annealing_dimod

        problem = X if hasattr(X, "to_model") else _problem_from_context(self)
        return run_annealing_dimod(problem, self.params)


@register_builtin("qudas.algorithm.qaoa_qiskit")
class QaoaQiskitStep(OptimizerStep):
    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = dict(params)

    def optimize(self, X=None, y=None):
        from qudas.algorithm.runners import run_qaoa_qiskit

        problem = X if getattr(X, "problem_id", None) == "maxcut" else _problem_from_context(self)
        return run_qaoa_qiskit(problem, self.params)


@register_builtin("qudas.passthrough")
class PassthroughStep(QdBaseStep):
    """テスト用: 入力をそのまま返す。"""

    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = dict(params)

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        return X

    def optimize(self, X=None, y=None):
        return X
