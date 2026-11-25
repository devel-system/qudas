"""MaxCut を Classical / Annealing / QAOA で解き、結果を比較するサンプル。

動作確認済みの最小例。有料 QPU は使用しません。
"""

from __future__ import annotations

from qudas.problem import MaxCutProblem
from qudas.api import solve, compare
from qudas.algorithm import AlgorithmPackage, get_algorithm
from qudas.qplat import QplatContract


def main() -> None:
    problem = MaxCutProblem.from_edge_list(
        [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
    )
    print("nodes:", problem.nodes)
    print("optimal (classical):", problem.classical_brute_force())

    algorithms = ["classical.exact", "annealing.dimod_exact"]
    try:
        import qiskit_aer  # noqa: F401

        algorithms.append("qaoa.qiskit")
    except ImportError:
        print("qiskit-aer not installed; skip QAOA")

    table = compare(
        problem,
        algorithms=algorithms,
        p=1,
        shots=256,
        opt_steps=8,
        seed=42,
    )
    for row in table["comparison"]["rows"]:
        print(
            f"- {row['algorithm']}: objective={row['problem_objective']} "
            f"status={row['status']}"
        )

    # Algorithm Package の保存と、別問題への再利用
    pkg = AlgorithmPackage.from_spec(get_algorithm("annealing.dimod_exact"))
    pkg.save("maxcut_annealing_package.json")
    other = MaxCutProblem.from_edge_list([(0, 1), (1, 2), (2, 0)])
    reused = AlgorithmPackage.load("maxcut_annealing_package.json").run(other)
    print("reused on triangle:", reused.problem_objective)

    # QPLAT 向け宣言データ
    print("qplat algorithms:", [a["algorithm_id"] for a in QplatContract.list_algorithms()])


if __name__ == "__main__":
    main()
