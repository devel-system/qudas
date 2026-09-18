"""
ゲート Executor（Bell 状態）の手動スモーク用スクリプト（pytest 対象外）。

実行例::

    python examples/manual/gate_bell_smoke.py
"""

from qudas.gate import QdGateBlock, QdGateExecutor, QdGateIR, QdGateInput


def main() -> None:
    gates = [
        QdGateIR(gate="h", targets=[0]),
        QdGateIR(gate="cx", targets=[1], controls=[0]),
    ]
    block = QdGateBlock(label="bell", gates=gates, num_qubits=2)
    qd_input = QdGateInput(blocks=[block])
    output = QdGateExecutor(provider="default").run(qd_input)
    print(output.results["bell"])


if __name__ == "__main__":
    main()
