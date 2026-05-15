import unittest

from qudas.annealing import (
    QdAnnealingBlock,
    QdAnnealingExecutor,
    QdAnnealingInput,
    QdAnnealingIR,
)


class TestAnnealingExecutor(unittest.TestCase):
    """アニーリング系モジュールの統合テスト。"""

    def setUp(self):
        self.qubo = {('q0', 'q0'): 1, ('q0', 'q1'): -1, ('q1', 'q1'): 2}
        self.ir = QdAnnealingIR(self.qubo)

    def test_pure_qudas_execution(self):
        block = QdAnnealingBlock(self.qubo, label='block0')
        qd_input = QdAnnealingInput([block])
        executor = QdAnnealingExecutor(provider='default')
        output = executor.run(qd_input)
        self.assertIn('block0', output.results)
        self.assertIn('solution', output.results['block0'])
        self.assertIn('energy', output.results['block0'])

    def test_qudas_to_dimod_execution(self):
        try:
            import dimod
        except ImportError:
            self.skipTest('dimod がインストールされていないためスキップ')

        bqm = self.ir.to_dimod_bqm()
        sampler = dimod.ExactSolver()
        sampleset = sampler.sample(bqm)
        best = sampleset.first
        self.assertIsNotNone(best)
        self.assertIsInstance(best.sample, dict)

    def test_dimod_to_qudas_execution(self):
        try:
            import dimod  # noqa: F401
        except ImportError:
            self.skipTest('dimod がインストールされていないためスキップ')

        bqm = self.ir.to_dimod_bqm()
        ir_from_dimod = QdAnnealingIR().from_dimod_bqm(bqm)
        block = QdAnnealingBlock(ir_from_dimod, label='block0')
        qd_input = QdAnnealingInput([block])
        executor = QdAnnealingExecutor(provider='default')
        output = executor.run(qd_input)
        self.assertIn('block0', output.results)
        self.assertIn('solution', output.results['block0'])
        self.assertIn('energy', output.results['block0'])

    def test_networkx_to_dimod_execution(self):
        try:
            import networkx as nx
            import dimod
        except ImportError:
            self.skipTest(
                'networkx または dimod がインストールされていないためスキップ'
            )

        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1, weight=-1)
        ir_from_nx = QdAnnealingIR().from_networkx(G)
        bqm = ir_from_nx.to_dimod_bqm()
        sampler = dimod.ExactSolver()
        sampleset = sampler.sample(bqm)
        self.assertTrue(len(sampleset) > 0)

    def test_parallel_qudas_execution(self):
        try:
            import dimod  # noqa: F401
        except ImportError:
            self.skipTest('dimod がインストールされていないためスキップ')

        qubo2 = {('q0', 'q0'): -1, ('q0', 'q1'): 2, ('q1', 'q1'): 1}
        blocks = [
            QdAnnealingBlock(self.qubo, label='block0'),
            QdAnnealingBlock(qubo2, label='block1'),
        ]
        qd_input = QdAnnealingInput(blocks)
        executor = QdAnnealingExecutor(provider_map={'block0': 'dimod'})
        output = executor.run_split(qd_input)
        self.assertIn('block0', output.results)
        self.assertIn('block1', output.results)
        for label in ['block0', 'block1']:
            self.assertIn('solution', output.results[label])
            self.assertIn('energy', output.results[label])


if __name__ == '__main__':
    unittest.main()
