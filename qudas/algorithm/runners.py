"""Algorithm 実行ディスパッチ（登録 runner）。"""

from __future__ import annotations

from typing import Any, Callable, Dict

from qudas.result.common import QdExperimentResult, Timer, TimingBreakdown, ResourceInfo

RunnerFn = Callable[..., QdExperimentResult]
_RUNNERS: Dict[str, RunnerFn] = {}


def register_runner(runner_id: str):
    def deco(fn: RunnerFn) -> RunnerFn:
        _RUNNERS[runner_id] = fn
        return fn

    return deco


def run_algorithm(
    runner_id: str,
    problem: Any,
    params: Dict[str, Any],
    package: Any = None,
) -> QdExperimentResult:
    if runner_id not in _RUNNERS:
        raise KeyError(
            f"No runner for {runner_id!r}. Available: {sorted(_RUNNERS)}"
        )
    return _RUNNERS[runner_id](problem, params, package=package)


def _ensure_compatible(problem: Any, compatible: list) -> None:
    pid = getattr(problem, "problem_id", None)
    if compatible and pid not in compatible and "*" not in compatible:
        raise ValueError(
            f"Problem {pid!r} is not compatible with this algorithm "
            f"(allowed={compatible})"
        )


@register_runner("classical.exact")
def run_classical_exact(problem, params, package=None) -> QdExperimentResult:
    _ensure_compatible(problem, ["maxcut", "knapsack", "tsp", "binary_optimization"])
    timer = Timer()
    if not hasattr(problem, "classical_brute_force"):
        raise TypeError("problem does not support classical_brute_force")
    out = problem.classical_brute_force()
    sol = out["solution"]
    obj = out.get("objective", problem.evaluate(sol))
    cons = (
        problem.evaluate_constraints(sol)
        if hasattr(problem, "evaluate_constraints")
        else {"feasible": out.get("feasible", True)}
    )
    return QdExperimentResult(
        solution=sol,
        objective_value=obj,
        problem_objective=obj,
        feasibility=cons.get("feasible", out.get("feasible", True)),
        timing=TimingBreakdown(total_s=timer.elapsed(), classical_optimize_s=timer.elapsed()),
        backend="classical",
        algorithm="classical.exact",
        parameters=dict(params),
        resources=ResourceInfo(num_variables=len(getattr(problem, "variables", []))),
        sense=getattr(problem, "sense", "maximize"),
        provenance={"package": getattr(package, "algorithm_id", None)},
    )


@register_runner("annealing.dimod_exact")
def run_annealing_dimod(problem, params, package=None) -> QdExperimentResult:
    _ensure_compatible(problem, ["maxcut", "knapsack", "tsp", "binary_optimization", "*"])
    timer = Timer()
    try:
        from qudas.annealing import QdAnnealingIR, QdAnnealingInput, QdAnnealingBlock, QdAnnealingExecutor
    except Exception as exc:  # pragma: no cover
        return QdExperimentResult(
            status="error",
            message=f"annealing stack unavailable: {exc}",
            algorithm="annealing.dimod_exact",
            backend="dimod",
        )

    qubo = problem.to_model("qubo")
    ir = QdAnnealingIR(qubo=qubo)
    block = QdAnnealingBlock(qubo=dict(ir.qubo), label="block0")
    inp = QdAnnealingInput(blocks=[block])
    executor = QdAnnealingExecutor(provider="dimod")
    raw_out = executor.run(inp)
    block_res = raw_out.results.get("block0") or next(iter(raw_out.results.values()))
    raw_sol = block_res.get("solution", {})
    energy = block_res.get("energy")
    reeval = problem.reevaluate(raw_sol)
    return QdExperimentResult(
        solution=reeval["solution"],
        objective_value=reeval["objective"],
        problem_objective=reeval["objective"],
        minimized_energy=float(energy) if energy is not None else None,
        feasibility=reeval["feasible"],
        timing=TimingBreakdown(total_s=timer.elapsed(), anneal_s=timer.elapsed()),
        backend="dimod.ExactSolver",
        algorithm="annealing.dimod_exact",
        parameters=dict(params),
        resources=ResourceInfo(num_variables=len(getattr(problem, "variables", []))),
        sense=getattr(problem, "sense", "maximize"),
        raw={"annealing_output": raw_out.to_dict() if hasattr(raw_out, "to_dict") else {}},
        provenance={"package": getattr(package, "algorithm_id", None)},
    )


@register_runner("qaoa.qiskit")
def run_qaoa_qiskit(problem, params, package=None) -> QdExperimentResult:
    """簡易 QAOA（MaxCut 向け）。論文の完全再現ではないサンプル実装。"""
    _ensure_compatible(problem, ["maxcut"])
    timer = Timer()
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        import numpy as np
    except ImportError as exc:
        return QdExperimentResult(
            status="error",
            message=f"qiskit/qiskit-aer required for QAOA: {exc}",
            algorithm="qaoa.qiskit",
            backend="qiskit",
        )

    if getattr(problem, "problem_id", None) != "maxcut":
        raise ValueError("qaoa.qiskit sample runner currently supports MaxCut only")

    p = int(params.get("p", 1))
    shots = int(params.get("shots", 512))
    seed = params.get("seed")
    reps = int(params.get("opt_steps", 30))
    lr = float(params.get("lr", 0.4))

    n = len(problem.nodes)
    edges = problem.edges

    def build_circuit(gamma, beta) -> QuantumCircuit:
        qc = QuantumCircuit(n, n)
        for i in range(n):
            qc.h(i)
        for layer in range(p):
            g = float(gamma[layer])
            b = float(beta[layer])
            for u, v, w in edges:
                # ZZ interaction for MaxCut cost
                qi = list(problem.nodes).index(u)
                qj = list(problem.nodes).index(v)
                qc.cx(qi, qj)
                qc.rz(2.0 * g * w, qj)
                qc.cx(qi, qj)
            for i in range(n):
                qc.rx(2.0 * b, i)
        qc.measure(range(n), range(n))
        return qc

    def bitstring_to_sol(bits: str) -> Dict[str, int]:
        # Qiskit little-endian: bits[0] is qubit n-1
        rev = bits[::-1]
        return {f"x{problem.nodes[i]}": int(rev[i]) for i in range(n)}

    def expected_cut(counts) -> float:
        total = sum(counts.values())
        acc = 0.0
        for bits, c in counts.items():
            sol = bitstring_to_sol(bits)
            acc += problem.cut_value(sol) * c / total
        return acc

    rng = np.random.default_rng(seed)
    gamma = rng.random(p) * np.pi
    beta = rng.random(p) * np.pi
    sim = AerSimulator()

    best_obj = float("-inf")
    best_sol = None
    best_counts = None

    for _ in range(reps):
        qc = build_circuit(gamma, beta)
        job = sim.run(qc, shots=shots, seed_simulator=seed)
        counts = job.result().get_counts()
        obj = expected_cut(counts)
        # 数値微分の簡易更新
        eps = 1e-1
        grad_g = np.zeros(p)
        grad_b = np.zeros(p)
        for i in range(p):
            g2 = gamma.copy()
            g2[i] += eps
            c2 = sim.run(build_circuit(g2, beta), shots=shots).result().get_counts()
            grad_g[i] = (expected_cut(c2) - obj) / eps
            b2 = beta.copy()
            b2[i] += eps
            c3 = sim.run(build_circuit(gamma, b2), shots=shots).result().get_counts()
            grad_b[i] = (expected_cut(c3) - obj) / eps
        gamma = gamma + lr * grad_g
        beta = beta + lr * grad_b
        if obj > best_obj:
            best_obj = obj
            best_counts = counts
            # 最頻ビット列
            top = max(counts.items(), key=lambda kv: kv[1])[0]
            best_sol = bitstring_to_sol(top)

    # 最終解を問題目的で再評価
    assert best_sol is not None
    final_obj = problem.evaluate(best_sol)
    gate_est = n + p * (3 * len(edges) + n)  # 粗い見積
    return QdExperimentResult(
        solution=best_sol,
        objective_value=final_obj,
        problem_objective=final_obj,
        feasibility=True,
        sampling_distribution=best_counts,
        timing=TimingBreakdown(
            total_s=timer.elapsed(),
            classical_optimize_s=timer.elapsed(),
            qpu_s=None,
        ),
        backend="qiskit_aer",
        algorithm="qaoa.qiskit",
        parameters={"p": p, "shots": shots, "seed": seed, "opt_steps": reps, "lr": lr},
        random_seed=seed,
        resources=ResourceInfo(
            num_qubits=n,
            shots=shots,
            gate_count=gate_est,
            circuit_depth=None,
        ),
        sense="maximize",
        message=(
            "Sample QAOA implementation for MaxCut; "
            "not a faithful reproduction of a specific paper."
        ),
        provenance={
            "package": getattr(package, "algorithm_id", None),
            "note": "educational/sample",
        },
    )
