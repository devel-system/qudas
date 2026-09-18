"""
ゲート出力の形式変換サンプル（pytest 対象外）。

流れ: 実行結果 → QdGateOutput / from_sdk_format / to_sdk_format / to_dict

実行例::

    python examples/manual/gate_output_convert.py
"""

from qudas.gate import (
    QdGateBlock,
    QdGateExecutor,
    QdGateInput,
    QdGateIR,
    QdGateOutput,
)


def main() -> None:
    gates = [
        QdGateIR(gate="h", targets=[0]),
        QdGateIR(gate="cx", targets=[1], controls=[0]),
    ]
    block = QdGateBlock(label="bell", gates=gates, num_qubits=2)
    qd_input = QdGateInput(blocks=[block])
    output = QdGateExecutor(provider="default").run(qd_input)

    print("=== QdGateOutput.results (executor) ===")
    print(output.results["bell"])

    sdk_obj = {
        "results": {
            "block0": {
                "counts": {"00": 512, "11": 512},
                "shots": 1024,
                "device": "qiskit_simulator",
            }
        }
    }
    qdo = QdGateOutput.from_sdk_format(sdk_obj, target="qiskit")
    print("=== from_sdk_format ===")
    print(f"{qdo.results=}")
    print("=== statistics ===")
    print(qdo.results["block0"].get("statistics"))

    print("=== to_sdk_format ===")
    print(qdo.to_sdk_format(target="qiskit"))

    print("=== to_dict ===")
    print(qdo.to_dict())

    restored = QdGateOutput.from_sdk_format(
        {"results": qdo.to_dict()},
        target="qiskit",
    )
    print("=== round-trip results['block0']['counts'] ===")
    print(restored.results["block0"]["counts"])


if __name__ == "__main__":
    main()
