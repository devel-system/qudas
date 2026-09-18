"""
ゲート入力の形式変換サンプル（pytest 対象外）。

流れ: Qiskit QuantumCircuit / OpenQASM → QdAlgorithmIR → QdGateBlock → QdGateInput

実行例::

    python examples/manual/gate_input_convert.py
"""

from qiskit import QuantumCircuit, qasm2

from qudas.gate import QdGateBlock, QdGateInput
from qudas.gate.ir import QdAlgorithmIR


def main() -> None:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(range(2), range(2))

    print("=== original QuantumCircuit ===")
    print(qc)

    ir = QdAlgorithmIR.from_qasm(qc)
    print("=== QdAlgorithmIR.gates ===")
    for g in ir.gates:
        print(
            f"  gate={g.gate}, targets={g.targets}, "
            f"controls={g.controls}, params={g.params}"
        )

    block = QdGateBlock(label="block0", gates=ir.gates, num_qubits=2)
    qd_input = QdGateInput(blocks=[block])
    print("=== QdGateInput.to_dict() ===")
    print(qd_input.to_dict())

    flat_ir = qd_input.to_ir()
    print(f"=== QdGateInput.to_ir() num_gates={len(flat_ir.gates)} ===")

    qasm_str = qasm2.dumps(qc)
    ir_from_qasm = QdAlgorithmIR.from_qasm(qasm_str)
    print(f"=== from OpenQASM string num_gates={len(ir_from_qasm.gates)} ===")

    qc_converted = ir.to_qiskit()
    print("=== converted QuantumCircuit (to_qiskit) ===")
    print(qc_converted)


if __name__ == "__main__":
    main()
