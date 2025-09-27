import unittest
from qudas.qudata import QuData

# その他必要なパッケージ
from amplify import VariableGenerator, Model, Poly, FixstarsClient, solve
from pyqubo import Binary, Base
from pyqubo.utils.asserts import assert_qubo_equal
import numpy as np
import pandas as pd
import pulp
import networkx as nx
import dimod
from sympy import Symbol, symbols, lambdify
from scipy.optimize import minimize, Bounds
from datetime import timedelta
from dotenv import load_dotenv
import os

# 環境変数の読み込み
load_dotenv()


def dicts_are_equal(dict1, dict2):
    """辞書のキーの順序を無視して等価性を比較する関数"""
    if len(dict1) != len(dict2):
        return False

    for k1, v1 in dict1.items():
        found = False
        for k2, v2 in dict2.items():
            if set(k1) == set(k2) and v1 == v2:
                found = True
                break
        if not found:
            return False
    return True


class TestQudata(unittest.TestCase):

    ##############################################
    # QuDataInput
    ##############################################
    def test_init_with_dict(self):
        """辞書データで初期化する場合のテスト"""
        prob = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        qudata = QuData.input(prob)
        self.assertTrue(dicts_are_equal(qudata.prob, prob))

    def test_init_with_none(self):
        """Noneで初期化する場合のテスト"""
        qudata = QuData.input()
        self.assertEqual(qudata.prob, {})

    def test_init_with_invalid_type(self):
        """無効な型で初期化しようとした場合のテスト"""
        with self.assertRaises(TypeError):
            QuData.input(123)  # 整数で初期化しようとした場合はTypeErrorが発生

    def test_add(self):
        """__add__メソッドのテスト"""
        prob1 = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        prob2 = {('q0', 'q0'): 2.0, ('q1', 'q1'): -1.0}
        qudata1 = QuData.input(prob1)
        qudata2 = QuData.input(prob2)
        result = qudata1 + qudata2
        expected = {
            ('q0', 'q1'): 1.0,
            ('q2', 'q2'): -1.0,
            ('q0', 'q0'): 2,
            ('q1', 'q1'): -1,
        }
        self.assertTrue(dicts_are_equal(result.prob, expected))

    def test_sub(self):
        """__sub__メソッドのテスト"""
        prob1 = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        prob2 = {('q0', 'q0'): 2.0, ('q1', 'q1'): -1.0}
        qudata1 = QuData.input(prob1)
        qudata2 = QuData.input(prob2)
        result = qudata1 - qudata2
        expected = {
            ('q0', 'q1'): 1.0,
            ('q2', 'q2'): -1.0,
            ('q0', 'q0'): -2,
            ('q1', 'q1'): 1,
        }
        self.assertTrue(dicts_are_equal(result.prob, expected))

    def test_mul(self):
        """__mul__メソッドのテスト"""
        prob1 = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        prob2 = {('q0', 'q0'): 2.0, ('q1', 'q1'): -1.0}
        qudata1 = QuData.input(prob1)
        qudata2 = QuData.input(prob2)
        result = qudata1 * qudata2
        expected = {('q0', 'q1'): 1.0, ('q0', 'q2'): -2.0, ('q1', 'q2'): 1.0}
        self.assertTrue(dicts_are_equal(result.prob, expected))

    def test_pow(self):
        """__pow__メソッドのテスト"""
        prob = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        qudata = QuData.input(prob)
        result = qudata**2
        expected = {('q0', 'q1'): 1.0, ('q0', 'q2', 'q1'): -2.0, ('q2', 'q2'): 1.0}
        self.assertTrue(dicts_are_equal(result.prob, expected))

    def test_pow_invalid_type(self):
        """__pow__で無効な型を渡した場合のテスト"""
        qudata = QuData.input({('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0})
        with self.assertRaises(TypeError):
            qudata ** 'invalid'  # 文字列を渡すとTypeErrorが発生する

    def test_from_pulp(self):
        """from_pulpメソッドのテスト"""

        # 変数の定義
        q0 = pulp.LpVariable('q0', lowBound=0, upBound=1, cat='Binary')
        q1 = pulp.LpVariable('q1', lowBound=0, upBound=1, cat='Binary')

        # 問題の定義 (2q0-q1)
        problem = pulp.LpProblem('QUBO', pulp.LpMinimize)
        problem += 2 * q0 - q1

        # QuData.inputオブジェクトを作成し、pulp問題を渡す
        qudata = QuData.input().from_pulp(problem)
        expected = {('q0', 'q0'): 2, ('q1', 'q1'): -1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_pulp_invalid_type(self):
        """from_pulpメソッドで無効な型のデータを渡した場合のテスト"""
        qudata = QuData.input()
        with self.assertRaises(TypeError):
            qudata.from_pulp("invalid")  # 無効な型でTypeErrorが発生するか確認

    def test_from_amplify(self):
        """from_amplifyメソッドのテスト"""
        # amplifyの設定
        q = VariableGenerator().array("Binary", shape=(3))
        objective = q[0] * q[1] - q[2] + 20

        # QuData.inputオブジェクトを作成し、amplify問題を渡す
        qudata = QuData.input().from_amplify(objective)
        expected = {('q_0', 'q_1'): 1.0, ('q_2', 'q_2'): -1.0}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_amplify_invalid_type(self):
        """from_amplifyメソッドで無効な型のデータを渡した場合のテスト"""
        qudata = QuData.input()
        with self.assertRaises(TypeError):
            qudata.from_amplify("invalid")  # 無効な型でTypeErrorが発生するか確認

    def test_from_pyqubo(self):
        """from_pyquboメソッドのテスト"""
        # pyquboのBaseのセットアップ
        q0, q1 = Binary("q0"), Binary("q1")
        prob = (q0 + q1) ** 2

        # QuData.inputオブジェクトを作成し、pyqubo問題を渡す
        qudata = QuData.input().from_pyqubo(prob)
        expected = {('q0', 'q0'): 1.0, ('q0', 'q1'): 2.0, ('q1', 'q1'): 1.0}
        self.assertEqual(qudata.prob, expected)

    def test_from_pyqubo_invalid_type(self):
        """from_pyquboメソッドで無効な型のデータを渡した場合のテスト"""
        qudata = QuData.input()
        with self.assertRaises(TypeError):
            qudata.from_pyqubo("invalid")  # 無効な型でTypeErrorが発生するか確認

    def test_from_array(self):
        """from_arrayメソッドのテスト"""
        # numpy配列のセットアップ
        prob = np.array(
            [
                [1, 1, 0],
                [0, 2, 0],
                [0, 0, -1],
            ]
        )

        # QuData.inputオブジェクトを作成し、配列を渡す
        qudata = QuData.input().from_array(prob)
        expected = {
            ('q_0', 'q_0'): 1,
            ('q_0', 'q_1'): 1,
            ('q_1', 'q_1'): 2,
            ('q_2', 'q_2'): -1,
        }
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_array_invalid_type(self):
        """from_arrayメソッドで無効な型のデータを渡した場合のテスト"""
        qudata = QuData.input()
        with self.assertRaises(TypeError):
            qudata.from_array("invalid")  # 無効な型でTypeErrorが発生するか確認

    def test_from_csv(self):
        """from_csvメソッドのテスト"""
        csv_file_path = './data/qudata.csv'
        qudata = QuData.input().from_csv(csv_file_path)
        expected = {
            ('q_0', 'q_0'): 1.0,
            ('q_0', 'q_2'): 2.0,
            ('q_1', 'q_1'): -1.0,
            ('q_2', 'q_1'): 2.0,
            ('q_2', 'q_2'): 2.0,
        }
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_csv_invalid(self):
        """from_csvメソッドで無効な型のデータを渡した場合のテスト"""
        invalid_csv_file_path = './data/invalid_data.csv'
        qudata = QuData.input()
        with self.assertRaises(ValueError, msg="読み取りエラー"):
            qudata.from_csv(invalid_csv_file_path)

    def test_from_json(self):
        """from_jsonメソッドのテスト"""
        json_file_path = './data/qudata.json'
        qudata = QuData.input().from_json(json_file_path)
        expected = {
            ('q0', 'q0'): 1.0,
            ('q0', 'q1'): 1.0,
            ('q1', 'q1'): -1.0,
            ('q2', 'q2'): 2.0,
        }
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_json_invalid(self):
        """from_jsonメソッドで無効な型のデータを渡した場合のテスト"""
        invalid_json_file_path = './data/invalid_data.json'
        qudata = QuData.input()
        with self.assertRaises(ValueError, msg="読み取りエラー"):
            qudata.from_json(invalid_json_file_path)

    def test_from_networkx(self):
        """from_networkxメソッドのテスト"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (0, 2)])
        qudata = QuData.input().from_networkx(G)
        expected = {('q_0', 'q_1'): 1, ('q_1', 'q_2'): 1, ('q_0', 'q_2'): 1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_networkx_invalid(self):
        """from_networkxメソッドで無効な型のデータを渡した場合のテスト"""
        qudata = QuData.input()
        with self.assertRaises(TypeError):
            qudata.from_amplify("invalid")  # 無効な型でTypeErrorが発生するか確認

    def test_from_pandas(self):
        """from_pandasメソッドのテスト"""
        array = np.array(
            [
                [1, 1, 0],
                [0, 2, 0],
                [0, 0, -1],
            ]
        )
        df = pd.DataFrame(array, columns=['q0', 'q1', 'q2'], index=['q0', 'q1', 'q2'])
        qudata = QuData.input().from_pandas(df)
        expected = {('q0', 'q0'): 1, ('q0', 'q1'): 1, ('q1', 'q1'): 2, ('q2', 'q2'): -1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_pandas_invalid(self):
        """from_pandasメソッドで無効な型のデータを渡した場合のテスト"""
        qudata = QuData.input()
        with self.assertRaises(TypeError):
            qudata.from_pandas("invalid")  # 無効な型でTypeErrorが発生するか確認

    def test_from_dimod_bqm(self):
        """from_dimod_bqmメソッドのテスト"""
        bqm = dimod.BinaryQuadraticModel(
            {'q2': -1}, {('q0', 'q1'): 1}, vartype='BINARY'
        )
        qudata = QuData.input().from_dimod_bqm(bqm)
        expected = {('q0', 'q1'): 1, ('q2', 'q2'): -1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_dimod_bqm_invalid(self):
        """from_dimod_bqmメソッドで無効な型のデータを渡した場合のテスト"""
        qudata = QuData.input()
        with self.assertRaises(TypeError):
            qudata.from_dimod_bqm("invalid")  # 無効な型でTypeErrorが発生するか確認

    def test_from_sympy(self):
        """from_sympyメソッドのテスト"""
        q0_sympy = Symbol('q0')
        q1_sympy = Symbol('q1')
        q2_sympy = Symbol('q2')
        prob_sympy = q0_sympy * q1_sympy - q2_sympy**2
        qudata = QuData.input().from_sympy(prob_sympy)
        expected = {('q0', 'q1'): 1, ('q2', 'q2'): -1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_sympy_invalid(self):
        """from_sympyメソッドで無効な型のデータを渡した場合のテスト"""
        qudata = QuData.input()
        with self.assertRaises(TypeError):
            qudata.from_sympy("invalid")  # 無効な型でTypeErrorが発生するか確認

    # 以下から to_xxxx
    def test_to_pulp(self):
        """pulp形式に変換するメソッドのテスト"""
        qudata = QuData.input({('q0', 'q0'): 2, ('q1', 'q1'): -1})
        prob = qudata.to_pulp()
        self.assertIsInstance(prob, pulp.LpProblem)

        # 変数の定義
        q0 = pulp.LpVariable('q0', lowBound=0, upBound=1, cat='Binary')
        q1 = pulp.LpVariable('q1', lowBound=0, upBound=1, cat='Binary')

        # 問題の定義 (2q0-q1)
        problem = pulp.LpProblem('QUBO', pulp.LpMinimize)
        problem += 2 * q0 - q1

        # 目標関数の比較
        self.assertEqual(str(prob.objective), str(problem.objective))

        # 変数リストの比較
        self.assertEqual(
            [v.name for v in prob.variables()], [v.name for v in problem.variables()]
        )

    def test_to_amplify(self):
        """amplify形式に変換するメソッドのテスト"""
        qudata = QuData.input(
            {('q_0', 'q_1'): 1.0, ('q_2', 'q_2'): -1.0}
        )  # q_xでない場合はエラー
        prob = qudata.to_amplify()
        self.assertIsInstance(prob, Poly)

        # amplifyの設定
        q = VariableGenerator().array("Binary", shape=(3))
        objective = q[0] * q[1] - q[2]

        # 目標関数の比較
        self.assertEqual(str(prob), str(objective))

    def test_to_pyqubo(self):
        """pyqubo形式に変換するメソッドのテスト"""
        qudata = QuData.input({('q0', 'q0'): 1.0, ('q0', 'q1'): 2.0, ('q1', 'q1'): 1.0})
        prob = qudata.to_pyqubo()
        self.assertIsInstance(prob, Base)

        # pyquboのBaseのセットアップ
        q0, q1 = Binary("q0"), Binary("q1")
        objective = (q0 + q1) ** 2

        # qubo式への変換
        qubo1, _ = prob.compile().to_qubo()
        qubo2, _ = objective.compile().to_qubo()

        # 目標関数の比較
        assert_qubo_equal(qubo1, qubo2)

    def test_to_array(self):
        """numpy形式に変換するメソッドのテスト"""
        qudata = QuData.input(
            {
                ('q_0', 'q_0'): 1,
                ('q_0', 'q_1'): 1,
                ('q_1', 'q_1'): 2,
                ('q_2', 'q_2'): -1,
            }
        )
        prob = qudata.to_array()

        # numpy配列のセットアップ
        array = np.array(
            [
                [1, 1, 0],
                [0, 2, 0],
                [0, 0, -1],
            ]
        )

        np.testing.assert_array_equal(prob, array)

    def test_to_csv(self):
        """CSV形式に保存するメソッドのテスト"""
        filename = "test_qudata"
        qudata = QuData.input(
            {
                ('q0', 'q0'): 1,
                ('q0', 'q2'): 2,
                ('q1', 'q1'): -1,
                ('q2', 'q1'): 2,
                ('q2', 'q2'): 2,
            }
        )
        qudata.to_csv(name=filename)
        self.assertTrue(os.path.exists(f"{filename}.csv"))
        os.remove(f"{filename}.csv")  # テスト後にファイルを削除

    def test_to_json(self):
        """JSON形式に保存するメソッドのテスト"""
        filename = "test_qudata"
        qudata = QuData.input(
            {('q0', 'q0'): 1, ('q0', 'q1'): 1, ('q1', 'q1'): -1, ('q2', 'q2'): 2}
        )
        qudata.to_json(name=filename)
        self.assertTrue(os.path.exists(f"{filename}.json"))
        os.remove(f"{filename}.json")  # テスト後にファイルを削除

    def test_to_networkx(self):
        """networkx形式に変換するメソッドのテスト"""
        qudata = QuData.input({('q_0', 'q_1'): 1, ('q_1', 'q_2'): 1, ('q_0', 'q_2'): 1})
        G = qudata.to_networkx()
        self.assertIsInstance(G, nx.Graph)

        # networkxの設定
        H = nx.Graph()
        H.add_edges_from([(0, 1), (1, 2), (0, 2)])
        self.assertTrue(G, H)

    def test_to_pandas(self):
        """pandas形式に変換するメソッドのテスト"""
        qudata = QuData.input(
            {('q0', 'q0'): 1, ('q0', 'q1'): 1, ('q1', 'q1'): 2, ('q2', 'q2'): -1}
        )
        df = qudata.to_pandas()

        # pandasの設定
        array = np.array(
            [
                [1, 1, 0],
                [0, 2, 0],
                [0, 0, -1],
            ]
        )
        expected_df = pd.DataFrame(
            array, columns=['q0', 'q1', 'q2'], index=['q0', 'q1', 'q2'], dtype=float
        )
        pd.testing.assert_frame_equal(df, expected_df)

    def test_to_dimod_bqm(self):
        """dimodのBQM形式に変換するメソッドのテスト"""
        qudata = QuData.input({('q0', 'q1'): 1, ('q2', 'q2'): -1})
        bqm = qudata.to_dimod_bqm()
        self.assertIsInstance(bqm, dimod.BinaryQuadraticModel)

        # dimodのBQM形式の確認
        expected_bqm = dimod.BinaryQuadraticModel(
            {'q2': -1}, {('q0', 'q1'): 1}, vartype='BINARY'
        )

        self.assertEqual(bqm, expected_bqm)

    def test_to_sympy(self):
        """sympy形式に変換するメソッドのテスト"""
        qudata = QuData.input({('q0', 'q1'): 1, ('q2', 'q2'): -1})
        expr = qudata.to_sympy()

        # sympyのSymbol形式の確認
        q0_sympy = Symbol('q0')
        q1_sympy = Symbol('q1')
        q2_sympy = Symbol('q2')
        prob_sympy = q0_sympy * q1_sympy - q2_sympy

        self.assertEqual(expr, prob_sympy)

    ##############################################
    # QuDataOutput
    ##############################################
    def test_from_pulp(self):
        # PuLP問題を解く
        prob = pulp.LpProblem("Test Problem", pulp.LpMinimize)
        x = pulp.LpVariable('x', lowBound=0, upBound=1, cat='Binary')
        y = pulp.LpVariable('y', lowBound=0, upBound=1, cat='Binary')
        prob += 2 * x - y
        prob.solve()

        # QuDataOutputのインスタンスを生成し、from_pulpメソッドをテスト
        qdo = QuData.output().from_pulp(prob)

        # 期待される結果
        expected_result = {'variables': {'x': 0, 'y': 1}, 'objective': -1}

        # 検証
        self.assertEqual(qdo.result, expected_result)
        self.assertEqual(qdo.result_type, 'pulp')

    def test_from_amplify(self):
        """Amplify形式の結果を受け取るメソッドのテスト"""
        try:
            from amplify import VariableGenerator, Model, FixstarsClient, solve
        except Exception:
            self.skipTest(
                "Amplifyがインストールされていないためテストをスキップします。"
            )

        from amplify import VariableGenerator, Model, FixstarsClient, solve
        from datetime import timedelta
        import os

        # Amplify形式の結果
        gen = VariableGenerator()
        q = gen.array("Binary", shape=(3))
        objective = 2 * q[0] - q[1] - q[2]

        # ソルバーの設定
        client = FixstarsClient()
        client.token = os.getenv("AMPLIFY_TOKEN")
        if client.token is None:
            self.skipTest(
                "Amplifyのトークンが設定されていないためテストをスキップします。"
            )

        client.parameters.timeout = timedelta(milliseconds=100)

        # 最小化を実行
        amplify_result = solve(Model(objective), client)

        # QuDataOutputのインスタンスを生成し、from_amplifyメソッドをテスト
        qdo = QuData.output().from_amplify(amplify_result)

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
        qdo = QuData.output().from_dimod(sampleset)

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
        qdo = QuData.output().from_scipy(res)

        # 期待される結果
        expected_result = {
            'variables': {'q0': 0.0, 'q1': 1.0, 'q2': 1.0},
            'objective': -2,
        }

        # 検証
        self.assertEqual(qdo.result, expected_result)
        self.assertEqual(qdo.result_type, 'scipy')

    def test_to_dimod(self):
        # Dimod形式の結果に変換
        qdo = QuData.output(
            result={'solution': {'q0': 0.0, 'q1': 1.0, 'q2': 1.0}, 'energy': -2}
        )
        dimod_result = qdo.to_dimod()

        # Dimod形式の結果
        qubo = {('q0', 'q0'): 2, ('q1', 'q1'): -1, ('q2', 'q2'): -1}
        sampleset = dimod.ExactSolver().sample_qubo(qubo)

        # 検証
        self.assertEqual(dimod_result.first, sampleset.first)

    def test_to_scipy(self):
        # SciPy形式の結果に変換
        qdo = QuData.output(
            result={'solution': {'q0': 0.0, 'q1': 1.0, 'q2': 1.0}, 'energy': -2}
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
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
