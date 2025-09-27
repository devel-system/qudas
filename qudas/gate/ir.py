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
            targets = [qc.qubits.index(q) for q in qargs]
            gate_ir = QdGateIR(
                gate=inst.name,
                targets=targets,
                controls=[],
                params=list(inst.parameters) if hasattr(inst, "parameters") else [],
            )
            gates.append(gate_ir)

        return cls(gates=gates)

    def to_qiskit(self):
        """保持している ``QdGateIR`` 一覧から ``qiskit.circuit.QuantumCircuit`` を生成する。"""
        from qiskit import QuantumCircuit  # type: ignore
        from qiskit.circuit import Instruction  # type: ignore
        from qiskit.circuit.library import XGate, HGate, CXGate, RZGate, RXGate, RYGate

        if not self.gates:
            return QuantumCircuit(0)

        # 回路に必要な量子ビット数を取得
        max_index = max(
            (
                max(g.targets + g.controls) if (g.targets or g.controls) else -1
                for g in self.gates
            )
        )
        num_qubits = max_index + 1

        # 測定ゲートがあるかどうか確認
        has_measure = any(g.gate == "measure" for g in self.gates)
        if has_measure:
            qc = QuantumCircuit(
                num_qubits, num_qubits
            )  # qubits と同数の classical bits
        else:
            qc = QuantumCircuit(num_qubits)

        # 名前→ゲートオブジェクトの対応表
        gate_map = {
            "x": XGate,
            "h": HGate,
            "cx": CXGate,
            "rz": RZGate,
            "rx": RXGate,
            "ry": RYGate,
            # 必要に応じて追加
        }

        for g in self.gates:
            # 制御 -> ターゲット の順で並べる
            qubit_indices = g.controls + g.targets

            if g.gate == "measure":
                # targets[0] を qubit, 同じ index の classical bit に対応付け
                for t in g.targets:
                    qc.measure(t, t)

            else:
                if g.gate in gate_map:
                    gate = (
                        gate_map[g.gate](*g.params) if g.params else gate_map[g.gate]()
                    )
                    qc.append(gate, [qc.qubits[i] for i in qubit_indices])
                else:
                    raise ValueError(f"Unsupported gate: {g.gate}")

        return qc
