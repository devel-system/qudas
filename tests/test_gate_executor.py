import unittest

from qudas.gate import (
    QdGateBlock,
    QdGateExecutor,
    QdGateIR,
    QdGateInput,
)


class TestGateExecutor(unittest.TestCase):
    """ゲート方式モジュールの統合テスト。"""

    def setUp(self):
        self.gates = [
            QdGateIR(gate='h', targets=[0]),
            QdGateIR(gate='cx', targets=[1], controls=[0]),
        ]
        self.block = QdGateBlock(label='block0', gates=self.gates, num_qubits=2)

    def test_pure_qudas_execution(self):
        qd_input = QdGateInput(blocks=[self.block])
        executor = QdGateExecutor(provider='default')
        output = executor.run(qd_input)
        self.assertIn('counts', output.results['block0'])
        self.assertIn('device', output.results['block0'])

    def test_qudas_to_qiskit_execution(self):
        try:
            from qiskit.primitives import Sampler
        except Exception:
            self.skipTest('qiskit がインストールされていないためスキップ')

        ir = self.block.to_ir()
        qc = ir.to_qiskit()
        qc.measure_all()
        sampler = Sampler()
        result = sampler.run([qc], shots=256).result()
        counts = result.quasi_dists[0]
        self.assertIsInstance(counts, dict)

    def test_qiskit_to_qudas_execution(self):
        try:
            from qiskit import QuantumCircuit
        except Exception:
            self.skipTest('qiskit がインストールされていないためスキップ')

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)

        from qudas.gate.ir import QdAlgorithmIR

        ir = QdAlgorithmIR.from_qasm(qc)
        num_qubits = 1
        block = QdGateBlock(label='block0', gates=ir.gates, num_qubits=num_qubits)
        qd_input = QdGateInput(blocks=[block])
        output = QdGateExecutor().run(qd_input)
        self.assertIn('counts', output.results['block0'])

    def test_qiskit_to_qasm_to_qiskit_execution(self):
        try:
            from qiskit import QuantumCircuit, qasm2
            from qiskit.primitives import Sampler
        except Exception:
            self.skipTest('qiskit がインストールされていないためスキップ')

        qc_original = QuantumCircuit(1, 1)
        qc_original.x(0)
        qc_original.measure(0, 0)
        qasm_str = qasm2.dumps(qc_original)

        from qudas.gate.ir import QdAlgorithmIR

        ir = QdAlgorithmIR.from_qasm(qasm_str)
        qc_converted = ir.to_qiskit()
        sampler = Sampler()
        result = sampler.run([qc_converted], shots=128).result()
        counts = result.quasi_dists[0]
        self.assertIsInstance(counts, dict)

    def test_parallel_run_split_execution(self):
        try:
            import qiskit  # noqa: F401
        except Exception:
            self.skipTest('qiskit がインストールされていないためスキップ')

        gates2 = [QdGateIR(gate='x', targets=[1])]
        block2 = QdGateBlock(label='block1', gates=gates2, num_qubits=2)
        qd_input = QdGateInput(blocks=[self.block, block2])
        executor = QdGateExecutor(provider_map={'block0': 'qiskit', 'block1': 'qiskit'})
        output = executor.run_split(qd_input)
        self.assertIn('block0', output.results)
        self.assertIn('block1', output.results)
        for label in ['block0', 'block1']:
            self.assertIn('counts', output.results[label])
            self.assertIn('device', output.results[label])


if __name__ == '__main__':
    unittest.main()
