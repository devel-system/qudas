"""
ゲート Executor（Grover 風サンプル）の手動スモーク用スクリプト（pytest 対象外）。

実行例::

    python examples/manual/gate_grover_smoke.py
"""

from qudas.gate import QdGateBlock, QdGateExecutor, QdGateIR, QdGateInput


def main() -> None:
    superposition = QdGateBlock(
        label="superposition",
        gates=[
            QdGateIR(gate="h", targets=[0]),
            QdGateIR(gate="h", targets=[1]),
        ],
        num_qubits=2,
    )
    oracle = QdGateBlock(
        label="oracle",
        gates=[QdGateIR(gate="cz", targets=[0, 1])],
        num_qubits=2,
    )
    diffusion = QdGateBlock(
        label="diffusion",
        gates=[
            QdGateIR(gate="h", targets=[0, 1]),
            QdGateIR(gate="x", targets=[0, 1]),
            QdGateIR(gate="cz", targets=[0, 1]),
            QdGateIR(gate="x", targets=[0, 1]),
            QdGateIR(gate="h", targets=[0, 1]),
        ],
        num_qubits=2,
    )
    grover_blocks = [superposition, oracle, diffusion]
    grover_input = QdGateInput(blocks=grover_blocks)

    executor = QdGateExecutor(
        provider="qiskit", provider_config={"backend": "qiskit_simulator"}
    )
    output = executor.run(grover_input)
    print(output.results)

    print("\n=== Raw results ===")
    print(output.to_dict())

    stats = output.results["superposition"]["statistics"]
    print("\n=== Statistics ===")
    print("Probability std :", stats["probability"]["std"])
    print("Unique bitstrings :", stats["bitstring"]["unique"])

    # output.visualize()


if __name__ == "__main__":
    main()
