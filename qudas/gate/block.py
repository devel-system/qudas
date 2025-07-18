from typing import List

from .gate_ir import QuantumGateIR


# 量子回路ブロック（SDK 非依存・構造表現用）
class QuantumCircuitBlock:
    """量子回路ブロック（SDK 非依存・構造表現用）。"""

    def __init__(self, name: str, gates: List[QuantumGateIR], num_qubits: int):
        self.name = name
        self.gates = gates
        self.num_qubits = num_qubits

    def to_ir(self):
        """IR として返す（簡易ユーティリティ）。"""
        from qudas.gate.ir import QuAlgorithmIR  # 遅延インポートで依存を最小化

        return QuAlgorithmIR(gates=self.gates)

    def __iter__(self):
        """`for gate in block` と書けるようにイテレータを実装。"""
        return iter(self.gates)

    def __repr__(self):
        return f"QuantumCircuitBlock(name={self.name!r}, num_qubits={self.num_qubits}, gates={len(self.gates)} ops)"


# Alias for backward compatibility / shorthand
QdBlock = QuantumCircuitBlock