"""
アニーリング Executor の手動スモーク用スクリプト（pytest 対象外）。

実行例::

    python examples/manual/annealing_smoke.py
"""

from qudas.annealing import QdAnnealingBlock, QdAnnealingExecutor, QdAnnealingInput
from qudas.annealing.output import QdAnnealingOutput


def main() -> None:
    qubo = {
        ("x0", "x0"): 1,
        ("x1", "x1"): 1,
        ("x2", "x2"): 2,
        ("x3", "x3"): 3,
        ("x0", "x1"): -3,
        ("x0", "x2"): 1,
        ("x0", "x3"): -1,
        ("x1", "x2"): 1,
        ("x1", "x3"): -1,
        ("x2", "x3"): -1,
    }
    block = QdAnnealingBlock(qubo, label="block0")
    anneal_input = QdAnnealingInput([block])
    executor = QdAnnealingExecutor(provider="dimod")
    output: QdAnnealingOutput = executor.run(anneal_input)

    print("=== Raw output ===")
    print(output.results)

    stats = output.results["block0"]["statistics"]
    print("\n=== Statistics ===")
    print("Energy std :", stats["energy"]["std"])
    print("Unique bitstrings :", stats["bitstring"]["unique"])

    # GUI 環境で確認する場合のみ有効化
    # output.visualize()


if __name__ == "__main__":
    main()
