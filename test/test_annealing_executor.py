import unittest

from qudas.annealing import (
    QdAnnealingIR,
    QdAnnealingBlock,
    QdAnnealingInput,
    QdAnnealingExecutor,
)


class TestAnnealingExecutor(unittest.TestCase):
    """アニーリング系モジュールの統合テスト。"""

    def setUp(self):
        # 簡単な QUBO を用意
        self.qubo = {('q0', 'q0'): 1, ('q0', 'q1'): -1, ('q1', 'q1'): 2}
        self.ir = QdAnnealingIR(self.qubo)

    # 1. 純粋な qudas での実行 -------------------------------------------------
    def test_pure_qudas_execution(self):
        """qudas での実行。"""
        block = QdAnnealingBlock(self.qubo, label='block0')
        qd_input = QdAnnealingInput([block])
        executor = QdAnnealingExecutor(provider='default')
        output = executor.run(qd_input)

        # 結果辞書の基本的なキーが存在するかを確認
        self.assertIn('block0', output.results)
        self.assertIn('solution', output.results['block0'])
        self.assertIn('energy', output.results['block0'])

    # 2. qudas -> 別フレームワーク (dimod) での実行 -----------------------------
    def test_qudas_to_dimod_execution(self):
        """qudas から dimod への実行。"""
        try:
            import dimod  # noqa: F401 – optional dependency
        except ImportError:
            self.skipTest('dimod がインストールされていないためスキップ')

        bqm = self.ir.to_dimod_bqm()
        sampler = dimod.ExactSolver()
        sampleset = sampler.sample(bqm)
        best = sampleset.first
        self.assertIsNotNone(best)
        self.assertIsInstance(best.sample, dict)

    # 3. 別フレームワーク (dimod) -> qudas での実行 ----------------------------
    def test_dimod_to_qudas_execution(self):
        """dimod から qudas への実行。"""
        try:
            import dimod  # noqa: F401
        except ImportError:
            self.skipTest('dimod がインストールされていないためスキップ')

        # dimod の BQM を作成
        bqm = self.ir.to_dimod_bqm()

        # qudas IR へ変換
        ir_from_dimod = QdAnnealingIR().from_dimod_bqm(bqm)

        # 実行
        block = QdAnnealingBlock(ir_from_dimod, label='block0')
        qd_input = QdAnnealingInput([block])
        executor = QdAnnealingExecutor(provider='default')
        output = executor.run(qd_input)

        self.assertIn('block0', output.results)
        self.assertIn('solution', output.results['block0'])
        self.assertIn('energy', output.results['block0'])

    # 4. 別フレームワーク (networkx) -> 別フレームワーク (dimod) での実行 --------
    def test_networkx_to_dimod_execution(self):
        """networkx グラフを作成し、qudas IR へ変換して dimod BQM へ変換し、ExactSolver で解く。"""
        try:
            import networkx as nx  # noqa: F401
            import dimod  # noqa: F401
        except ImportError:
            self.skipTest(
                'networkx または dimod がインストールされていないためスキップ'
            )

        import networkx as nx

        # networkx グラフを作成
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1, weight=-1)

        # networkx -> qudas IR
        ir_from_nx = QdAnnealingIR().from_networkx(G)

        # qudas IR -> dimod BQM
        bqm = ir_from_nx.to_dimod_bqm()
        sampler = dimod.ExactSolver()
        sampleset = sampler.sample(bqm)
        self.assertTrue(len(sampleset) > 0)

    # 5. 並列実行（複数ブロック） -------------------------------------------
    def test_parallel_qudas_execution(self):
        """2 ブロックを渡して Executor.run() が並列評価するケースをテスト。"""
        # 追加の QUBO を用意
        qubo2 = {('q0', 'q0'): -1, ('q0', 'q1'): 2, ('q1', 'q1'): 1}

        blocks = [
            QdAnnealingBlock(self.qubo, label='block0'),
            QdAnnealingBlock(qubo2, label='block1'),
        ]
        qd_input = QdAnnealingInput(blocks)

        # backend_map で block0 を dimod, block1 をデフォルトに設定してみる
        executor = QdAnnealingExecutor(provider_map={'block0': 'dimod'})
        output = executor.run_split(qd_input)

        # 両ブロックの結果が取得できているか
        self.assertIn('block0', output.results)
        self.assertIn('block1', output.results)
        for label in ['block0', 'block1']:
            self.assertIn('solution', output.results[label])
            self.assertIn('energy', output.results[label])


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
