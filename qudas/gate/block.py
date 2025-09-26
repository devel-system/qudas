from typing import List, Optional
from qudas.gate.gate_ir import QdGateIR


# 量子回路ブロック（SDK 非依存・構造表現用）
class QdGateBlock:
    """量子回路ブロック（SDK 非依存・構造表現用）。"""

    def __init__(
        self,
        gates: List[QdGateIR],
        num_qubits: Optional[int] = None,
        label: str = "block",
    ):
        """量子回路ブロック（SDK 非依存・構造表現用）。

        Parameters
        ----------
        gates : List[QdGateIR]
            ブロックに含まれるゲートのリスト。
        num_qubits : int, optional
            ブロックに含まれる量子ビットの数。
        label : str, optional
            ブロックのラベル。

        Raises:
            TypeError: gates に QdGateIR のリスト以外を渡した場合。
            TypeError: num_qubits に int 以外を渡した場合。
        """
        if not all(isinstance(gate, QdGateIR) for gate in gates):
            raise TypeError("gates には QdGateIR のリストを渡してください。")
        if num_qubits is None:
            num_qubits = self._infer_num_qubits(gates)
        elif not isinstance(num_qubits, int):
            raise TypeError("num_qubits には int を渡してください。")
        self.gates = gates
        self.num_qubits = num_qubits
        self.label = label

    def _infer_num_qubits(self, gates: List[QdGateIR]) -> int:
        max_index = -1
        for g in gates:
            for idx in g.targets + g.controls:  # 複数対応
                max_index = max(max_index, idx)
        return max_index + 1  # 0-indexed → qubit数は +1

    def to_ir(self):
        """IR として返す（簡易ユーティリティ）。"""
        from qudas.gate.ir import QdAlgorithmIR  # 遅延インポートで依存を最小化

        return QdAlgorithmIR(gates=self.gates)

    def __iter__(self):
        """`for gate in block` と書けるようにイテレータを実装。"""
        return iter(self.gates)

    def __repr__(self):
        return f"QdGateBlock(label={self.label!r}, num_qubits={self.num_qubits}, gates={len(self.gates)} ops)"


# Alias for backward compatibility / shorthand
QdBlock = QdGateBlock
