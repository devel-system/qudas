from qudas.gate.gate_ir import QdGateIR
from typing import List, Iterable

class QdAlgorithmIR:
    def __init__(self, gates: List[QdGateIR]):
        self.gates = gates

    @classmethod
    def from_blocks(cls, blocks: Iterable[Iterable[QdGateIR]]):
        """量子回路ブロックの集合から ``QdAlgorithmIR`` を生成する。

        Parameters
        ----------
        blocks : Iterable[Iterable[QdGateIR]]
            各ブロックが ``QdGateIR`` を要素にもつ反復可能オブジェクト。
            典型的には :class:`qudas.gate.block.QdGateBlock` のリストを想定。
        """

        gates: List[QdGateIR] = []
        for block in blocks:
            for gate in block:
                if isinstance(gate, QdGateIR):
                    gates.append(gate)
                else:
                    # 型が異なる場合はスキップ／将来拡張時に警告など
                    continue

        return cls(gates=gates)

    @classmethod
    def from_qasm(cls, qasm):
        """OpenQASM 文字列 / ファイルパス / QuantumCircuit から ``QdAlgorithmIR`` を生成する。
        (QdGateIR ベース)"""
        from qiskit import QuantumCircuit  # type: ignore
        import os

        if isinstance(qasm, QuantumCircuit):
            qc = qasm
        elif isinstance(qasm, str):
            if os.path.exists(qasm):
                qc = QuantumCircuit.from_qasm_file(qasm)
            else:
                qc = QuantumCircuit.from_qasm_str(qasm)
        else:
            raise TypeError(f"{type(qasm)} は対応していない型です。")

        gates: List[QdGateIR] = []
        for inst, qargs, _ in qc.data:
            targets = [q.index for q in qargs]
            gate_ir = QdGateIR(
                gate=inst.name,
                targets=targets,
                controls=[],
                params=list(inst.params) if inst.params is not None else [],
            )
            gates.append(gate_ir)

        return cls(gates=gates)

    def to_qiskit(self):
        """保持している ``QdGateIR`` 一覧から ``qiskit.circuit.QuantumCircuit`` を生成する。"""
        from qiskit import QuantumCircuit  # type: ignore
        from qiskit.circuit import Instruction  # type: ignore

        if not self.gates:
            return QuantumCircuit(0)

        # 回路に必要な量子ビット数を取得
        max_index = max(
            (
                max(g.targets + g.controls)
                if (g.targets or g.controls)
                else -1
                for g in self.gates
            )
        )
        num_qubits = max_index + 1
        qc = QuantumCircuit(num_qubits)

        for g in self.gates:
            # 制御 -> ターゲット の順で並べる
            qubit_indices = g.controls + g.targets
            instruction = Instruction(
                name=g.gate,
                num_qubits=len(qubit_indices),
                num_clbits=0,
                params=g.params,
            )
            qc.append(instruction, [qc.qubits[i] for i in qubit_indices])

        return qc