import unittest
from qudas.qudata import QuDataOutput

# その他必要なパッケージ
from amplify import VariableGenerator, Model, Poly, FixstarsClient, solve
from datetime import timedelta
import numpy as np
from pulp import LpVariable, LpProblem, LpMinimize, value
import dimod
from sympy import Symbol, symbols, lambdify
from scipy.optimize import minimize, Bounds


class TestQuDataOutput(unittest.TestCase):

    def test_from_pulp(self):
        # PuLP問題を解く
        prob = LpProblem("Test Problem", LpMinimize)
        x = LpVariable('x', lowBound=0, upBound=1, cat='Binary')
        y = LpVariable('y', lowBound=0, upBound=1, cat='Binary')
        prob += 2 * x - y
        prob.solve()

        # QuDataOutputのインスタンスを生成し、from_pulpメソッドをテスト
        qdo = QuDataOutput().from_pulp(prob)

        # 期待される結果
        expected_result = {'variables': {'x': 0, 'y': 1}, 'objective': -1}

        # 検証
        self.assertEqual(qdo.result, expected_result)
        self.assertEqual(qdo.result_type, 'pulp')

    def test_from_amplify(self):
        # Amplify形式の結果
        gen = VariableGenerator()
        q = gen.array("Binary", shape=(3))
        objective = 2 * q[0] - q[1] - q[2]

        # ソルバーの設定
        client = FixstarsClient()
        client.token = "AE/HaqGh1iuFMEennXk10xS1LCgld8D18oC"
        client.parameters.timeout = timedelta(milliseconds=100)

        # 最小化を実行
        amplify_result = solve(Model(objective), client)

        # QuDataOutputのインスタンスを生成し、from_amplifyメソッドをテスト
        qdo = QuDataOutput().from_amplify(amplify_result)

        # 期待される結果
        expected_result = {
            'variables': {'q_0': 0.0, 'q_1': 1.0, 'q_2': 1.0},
            'objective': -2,
        }

        # 検証
        self.assertEqual(qdo.result, expected_result)
        self.assertEqual(qdo.result_type, 'amplify')

    def test_from_dimod(self):
        # Dimod形式の結果
        qubo = {('q0', 'q0'): 2, ('q1', 'q1'): -1, ('q2', 'q2'): -1}
        sampleset = dimod.ExactSolver().sample_qubo(qubo)

        # QuDataOutputのインスタンスを生成し、from_dimodメソッドをテスト
        qdo = QuDataOutput().from_dimod(sampleset)

        # 期待される結果
        expected_result = {
            'variables': {'q0': 0.0, 'q1': 1.0, 'q2': 1.0},
            'objective': -2,
        }

        # 検証
        self.assertEqual(qdo.result, expected_result)
        self.assertEqual(qdo.result_type, 'dimod')

    def test_from_scipy(self):

        # シンボリック変数の定義
        q0, q1, q2 = symbols('q0 q1 q2')

        # 目的関数を定義
        objective_function = 2 * q0 - q1 - q2

        # シンボリック関数を数値化して評価できる形式に変換
        f = lambdify([q0, q1, q2], objective_function, 'numpy')

        # 初期解 (すべて0に設定)
        q = [0.5, 0.5, 0.5]

        # バイナリ変数の範囲を定義 (0 <= x <= 1)
        bounds = Bounds([0, 0, 0], [1, 1, 1])

        # 制約なしで最適化を行い、結果をバイナリ制約に丸める
        res = minimize(lambda q: f(q[0], q[1], q[2]), q, method='SLSQP', bounds=bounds)

        # QuDataOutputのインスタンスを生成し、from_scipyメソッドをテスト
        qdo = QuDataOutput().from_scipy(res)

        # 期待される結果
        expected_result = {
            'variables': {'q0': 0.0, 'q1': 1.0, 'q2': 1.0},
            'objective': -2,
        }

        # 検証
        self.assertEqual(qdo.result, expected_result)
        self.assertEqual(qdo.result_type, 'sympy')

    def test_to_dimod(self):
        # Dimod形式の結果に変換
        qdo = QuDataOutput(
            result={'variables': {'q0': 0.0, 'q1': 1.0, 'q2': 1.0}, 'objective': -2}
        )
        dimod_result = qdo.to_dimod()

        # Dimod形式の結果
        qubo = {('q0', 'q0'): 2, ('q1', 'q1'): -1, ('q2', 'q2'): -1}
        sampleset = dimod.ExactSolver().sample_qubo(qubo)

        # 検証
        self.assertEqual(dimod_result.first, sampleset.first)

    def test_to_scipy(self):
        # SciPy形式の結果に変換
        qdo = QuDataOutput(
            result={'variables': {'q0': 0.0, 'q1': 1.0, 'q2': 1.0}, 'objective': -2}
        )
        scipy_result = qdo.to_scipy()

        # シンボリック変数の定義
        q0, q1, q2 = symbols('q0 q1 q2')

        # 目的関数を定義
        objective_function = 2 * q0 - q1 - q2

        # シンボリック関数を数値化して評価できる形式に変換
        f = lambdify([q0, q1, q2], objective_function, 'numpy')

        # 初期解 (すべて0に設定)
        q = [0.5, 0.5, 0.5]

        # バイナリ変数の範囲を定義 (0 <= x <= 1)
        bounds = Bounds([0, 0, 0], [1, 1, 1])

        # 制約なしで最適化を行い、結果をバイナリ制約に丸める
        res = minimize(lambda q: f(q[0], q[1], q[2]), q, method='SLSQP', bounds=bounds)

        # 検証
        np.testing.assert_array_equal(scipy_result.x, res.x)
        self.assertEqual(scipy_result.fun, res.fun)
        self.assertEqual(scipy_result.success, res.success)


if __name__ == '__main__':
    unittest.main()
