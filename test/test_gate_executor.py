import unittest

from qudas.gate import (
    QdGateBlock,
    QdGateIR,
    QdGateInput,
    QdGateExecutor,
)


class TestGateExecutor(unittest.TestCase):
    """ゲート方式モジュールの統合テスト。"""

    def setUp(self):
        # 2 量子ビットで簡単な回路を用意 (H + CX)
        self.gates = [
            QdGateIR(gate='h', targets=[0]),
            QdGateIR(gate='cx', targets=[1], controls=[0]),
        ]
        self.block = QdGateBlock(label='block0', gates=self.gates, num_qubits=2)

    # 1. 純粋な qudas での実行 -------------------------------------------------
    def test_pure_qudas_execution(self):
        """qudas での実行。"""
        qd_input = QdGateInput(blocks=[self.block])
        executor = QdGateExecutor(provider='default')
        output = executor.run(qd_input)

        # 実行結果に counts と device 情報が含まれているか
        self.assertIn('counts', output.results['block0'])
        self.assertIn('device', output.results['block0'])

    # 2. qudas -> 外部フレームワーク (qiskit) での実行 ------------------------
    def test_qudas_to_qiskit_execution(self):
        """qudas から qiskit への実行。"""
        try:
            from qiskit.primitives import Sampler  # noqa: F401
        except Exception:
            self.skipTest('qiskit がインストールされていないためスキップ')

        ir = self.block.to_ir()
        qc = ir.to_qiskit()
        qc.measure_all()

        from qiskit.primitives import Sampler

        sampler = Sampler()
        result = sampler.run([qc], shots=256).result()
        counts = result.quasi_dists[0]
        self.assertIsInstance(counts, dict)

    # 3. 外部フレームワーク (qiskit) -> qudas での実行 -------------------------
    def test_qiskit_to_qudas_execution(self):
        """qiskit から qudas への実行。"""
        try:
            from qiskit import QuantumCircuit  # noqa: F401
        except Exception:
            self.skipTest('qiskit がインストールされていないためスキップ')

        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)

        # qiskit Circuit -> QuAlgorithmIR
        from qudas.gate.ir import QdAlgorithmIR

        ir = QdAlgorithmIR.from_qasm(qc)

        # QuAlgorithmIR -> QuantumCircuitBlock -> qudas 実行
        num_qubits = 1
        block = QdGateBlock(label='block0', gates=ir.gates, num_qubits=num_qubits)
        qd_input = QdGateInput(blocks=[block])
        output = QdGateExecutor().run(qd_input)
        self.assertIn('counts', output.results['block0'])

    # 4. 外部フレームワーク (qiskit) -> qasm 文字列 -> qiskit での実行 ----------
    def test_qiskit_to_qasm_to_qiskit_execution(self):
        """qiskit から qasm 文字列へ変換し、再度 qiskit への実行。"""
        try:
            from qiskit import QuantumCircuit, qasm2  # noqa: F401
            from qiskit.primitives import Sampler
        except Exception:
            self.skipTest('qiskit がインストールされていないためスキップ')

        from qiskit import QuantumCircuit, qasm2
        from qiskit.primitives import Sampler

        # 元となる回路
        qc_original = QuantumCircuit(1, 1)
        qc_original.x(0)
        qc_original.measure(0, 0)

        # qasm 文字列へ変換
        qasm_str = qasm2.dumps(qc_original)

        # qasm -> QuAlgorithmIR -> 再度 qiskit Circuit
        from qudas.gate.ir import QdAlgorithmIR

        ir = QdAlgorithmIR.from_qasm(qasm_str)
        qc_converted = ir.to_qiskit()

        sampler = Sampler()
        result = sampler.run([qc_converted], shots=128).result()
        counts = result.quasi_dists[0]
        self.assertIsInstance(counts, dict)

    # 5. 並列実行（run_split） -----------------------------------------------
    def test_parallel_run_split_execution(self):
        """2 ブロックを run_split() で並列実行するケースをテスト。"""
        # 2 つ目のブロックを追加 (簡易 X ゲートのみ)
        gates2 = [QdGateIR(gate='x', targets=[1])]
        block2 = QdGateBlock(label='block1', gates=gates2, num_qubits=2)

        qd_input = QdGateInput(blocks=[self.block, block2])

        executor = QdGateExecutor(provider_map={'block0': 'qiskit', 'block1': 'qiskit'})
        output = executor.run_split(qd_input)

        # 結果辞書に両ブロックが存在するか
        self.assertIn('block0', output.results)
        self.assertIn('block1', output.results)
        for label in ['block0', 'block1']:
            self.assertIn('counts', output.results[label])
            self.assertIn('device', output.results[label])


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
