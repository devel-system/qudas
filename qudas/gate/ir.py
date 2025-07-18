class QuAlgorithmIR:
    def __init__(self, gates):
        self.gates = gates

    @classmethod
    def from_blocks(cls, blocks):
        return cls(gates=[g for b in blocks for g in b])

    @classmethod
    def from_qasm(cls, qasm):
        """OpenQASM 文字列 / ファイルパス / QuantumCircuit から ``QuAlgorithmIR`` を生成する。

        Parameters
        ----------
        qasm : str | qiskit.circuit.QuantumCircuit
            以下のいずれかを受け付ける。

            * OpenQASM 形式の文字列
            * OpenQASM ファイルのパス (``.qasm`` 拡張子など)
            * ``qiskit.circuit.QuantumCircuit`` オブジェクト

        Returns
        -------
        QuAlgorithmIR
            パースした回路の各ゲートを `gates` リストとして保持するクラスインスタンス。
        """

        # lazy import : qiskit が未使用の場合の起動コストを抑える
        from qiskit import QuantumCircuit  # type: ignore
        import os

        # QuantumCircuit が渡された場合
        if isinstance(qasm, QuantumCircuit):
            qc = qasm

        # 文字列が渡された場合 (ファイルパス or QASM 文字列)
        elif isinstance(qasm, str):
            # ファイルパスとして存在する場合はファイル読み込みを優先
            if os.path.exists(qasm):
                qc = QuantumCircuit.from_qasm_file(qasm)
            else:
                # 直接 QASM 文字列として解釈
                qc = QuantumCircuit.from_qasm_str(qasm)

        else:
            raise TypeError(f"{type(qasm)} は対応していない型です。")

        # QuantumCircuit.data は (Instruction, qubits, clbits) のタプルからなるリスト
        # Instruction オブジェクトのみを保持して IR の gates とする。
        gates = [inst for inst, _qargs, _cargs in qc.data]

        return cls(gates=gates)


QdIR = QuAlgorithmIR