import unittest
from qudas import QuDataInput

# その他必要なパッケージ
from amplify import VariableGenerator, Poly
import numpy as np
import pandas as pd
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import dimod
from sympy import Symbol

def dicts_are_equal(dict1, dict2):
    """辞書のキーの順序を無視して等価性を比較する関数"""
    if len(dict1) != len(dict2):
        return False

    for (k1, v1) in dict1.items():
        found = False
        for (k2, v2) in dict2.items():
            if set(k1) == set(k2) and v1 == v2:
                found = True
                break
        if not found:
            return False
    return True

class TestQudata(unittest.TestCase):

    def test_init_with_dict(self):
        """辞書データで初期化する場合のテスト"""
        prob = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        qudata = QuDataInput(prob)
        self.assertTrue(dicts_are_equal(qudata.prob, prob))


    def test_init_with_none(self):
        """Noneで初期化する場合のテスト"""
        qudata = QuDataInput()
        self.assertEqual(qudata.prob, {})

    def test_init_with_invalid_type(self):
        """無効な型で初期化しようとした場合のテスト"""
        with self.assertRaises(TypeError):
            QuDataInput(123)  # 整数で初期化しようとした場合はTypeErrorが発生

    def test_add(self):
        """__add__メソッドのテスト"""
        prob1 = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        prob2 = {('q0', 'q0'): 2.0, ('q1', 'q1'): -1.0}
        qudata1 = QuDataInput(prob1)
        qudata2 = QuDataInput(prob2)
        result = qudata1 + qudata2
        expected = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0, ('q0', 'q0'): 2, ('q1', 'q1'): -1}
        self.assertTrue(dicts_are_equal(result.prob, expected))

    def test_sub(self):
        """__sub__メソッドのテスト"""
        prob1 = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        prob2 = {('q0', 'q0'): 2.0, ('q1', 'q1'): -1.0}
        qudata1 = QuDataInput(prob1)
        qudata2 = QuDataInput(prob2)
        result = qudata1 - qudata2
        expected = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0, ('q0', 'q0'): -2, ('q1', 'q1'): 1}
        self.assertTrue(dicts_are_equal(result.prob, expected))

    def test_mul(self):
        """__mul__メソッドのテスト"""
        prob1 = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        prob2 = {('q0', 'q0'): 2.0, ('q1', 'q1'): -1.0}
        qudata1 = QuDataInput(prob1)
        qudata2 = QuDataInput(prob2)
        result = qudata1 * qudata2
        expected = {('q0', 'q1'): 1.0, ('q0', 'q2'): -2.0, ('q1', 'q2'): 1.0}
        self.assertTrue(dicts_are_equal(result.prob, expected))

    def test_pow(self):
        """__pow__メソッドのテスト"""
        prob = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        qudata = QuDataInput(prob)
        result = qudata ** 2
        expected = {('q0', 'q1'): 1.0, ('q0', 'q2', 'q1'): -2.0, ('q2', 'q2'): 1.0}
        self.assertTrue(dicts_are_equal(result.prob, expected))

    def test_pow_invalid_type(self):
        """__pow__で無効な型を渡した場合のテスト"""
        qudata = QuDataInput({('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0})
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

        # QuDataInputオブジェクトを作成し、pulp問題を渡す
        qudata = QuDataInput().from_pulp(problem)
        expected = {('q0', 'q0'): 2, ('q1', 'q1'): -1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_pulp_invalid_type(self):
        """from_pulpメソッドで無効な型のデータを渡した場合のテスト"""
        qudata = QuDataInput()
        with self.assertRaises(TypeError):
            qudata.from_pulp("invalid")  # 無効な型でTypeErrorが発生するか確認

    def test_from_amplify(self):
        """from_amplifyメソッドのテスト"""
        # amplifyの設定
        q = VariableGenerator().array("Binary", shape=(3))
        objective = q[0] * q[1] - q[2]

        # QuDataInputオブジェクトを作成し、amplify問題を渡す
        qudata = QuDataInput().from_amplify(objective)
        expected = {('q_0', 'q_1'): 1.0, ('q_2', 'q_2'): -1.0}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_amplify_invalid_type(self):
        """from_amplifyメソッドで無効な型のデータを渡した場合のテスト"""
        qudata = QuDataInput()
        with self.assertRaises(TypeError):
            qudata.from_amplify("invalid")  # 無効な型でTypeErrorが発生するか確認

    # def test_from_pyqubo(self):
    #     """from_pyquboメソッドのテスト"""
    #     # pyquboのAddのセットアップ
    #     x = Array.create('x', shape=2)
    #     prob = (x[0] + x[1]) ** 2

    #     # QuDataInputオブジェクトを作成し、pyqubo問題を渡す
    #     qudata = QuDataInput().from_pyqubo(prob)
    #     expected = {('x[0]', 'x[0]'): 1.0, ('x[0]', 'x[1]'): 2.0, ('x[1]', 'x[1]'): 1.0}
    #     self.assertEqual(qudata.prob, expected)

    # def test_from_pyqubo_invalid_type(self):
    #     """from_pyquboメソッドで無効な型のデータを渡した場合のテスト"""
    #     qudata = QuDataInput()
    #     with self.assertRaises(TypeError):
    #         qudata.from_pyqubo("invalid")  # 無効な型でTypeErrorが発生するか確認

    def test_from_array(self):
        """from_arrayメソッドのテスト"""
        # numpy配列のセットアップ
        prob = np.array([
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, -1],
        ])

        # QuDataInputオブジェクトを作成し、配列を渡す
        qudata = QuDataInput().from_array(prob)
        expected = {('q_0', 'q_0'): 1, ('q_0', 'q_1'): 1, ('q_1', 'q_1'): 2, ('q_2', 'q_2'): -1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_array_invalid_type(self):
        """from_arrayメソッドで無効な型のデータを渡した場合のテスト"""
        qudata = QuDataInput()
        with self.assertRaises(TypeError):
            qudata.from_array("invalid")  # 無効な型でTypeErrorが発生するか確認

    def test_from_csv(self):
        """from_csvメソッドのテスト"""
        csv_file_path = './data/qudata.csv'
        qudata = QuDataInput().from_csv(csv_file_path)
        expected = {('q_0', 'q_0'): 1.0, ('q_0', 'q_2'): 2.0, ('q_1', 'q_1'): -1.0, ('q_2', 'q_1'): 2.0, ('q_2', 'q_2'): 2.0}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_csv_invalid(self):
        """from_csvメソッドで無効な型のデータを渡した場合のテスト"""
        invalid_csv_file_path = './data/invalid_data.csv'
        qudata = QuDataInput()
        with self.assertRaises(ValueError, msg="読み取りエラー"):
            qudata.from_csv(invalid_csv_file_path)

    def test_from_json(self):
        """from_jsonメソッドのテスト"""
        json_file_path = './data/qudata.json'
        qudata = QuDataInput().from_json(json_file_path)
        expected = {('q0', 'q0'): 1.0, ('q0', 'q1'): 1.0, ('q1', 'q1'): -1.0, ('q2', 'q2'): 2.0}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_json_invalid(self):
        """from_jsonメソッドで無効な型のデータを渡した場合のテスト"""
        invalid_json_file_path = './data/invalid_data.json'
        qudata = QuDataInput()
        with self.assertRaises(ValueError, msg="読み取りエラー"):
            qudata.from_json(invalid_json_file_path)

    def test_from_networkx(self):
        """from_networkxメソッドのテスト"""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (0, 2)])
        qudata = QuDataInput().from_networkx(G)
        expected = {('q_0', 'q_1'): 1, ('q_1', 'q_2'): 1, ('q_0', 'q_2'): 1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_networkx_invalid(self):
        """from_networkxメソッドで無効な型のデータを渡した場合のテスト"""
        qudata = QuDataInput()
        with self.assertRaises(TypeError):
            qudata.from_amplify("invalid")  # 無効な型でTypeErrorが発生するか確認

    def test_from_pandas(self):
        """from_pandasメソッドのテスト"""
        array = np.array([
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, -1],
        ])
        df = pd.DataFrame(array, columns=['q0', 'q1', 'q2'], index=['q0', 'q1', 'q2'])
        qudata = QuDataInput().from_pandas(df)
        expected = {('q0', 'q0'): 1, ('q0', 'q1'): 1, ('q1', 'q1'): 2, ('q2', 'q2'): -1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_pandas_invalid(self):
        """from_pandasメソッドで無効な型のデータを渡した場合のテスト"""
        qudata = QuDataInput()
        with self.assertRaises(TypeError):
            qudata.from_pandas("invalid")  # 無効な型でTypeErrorが発生するか確認

    def test_from_dimod_bqm(self):
        """from_dimod_bqmメソッドのテスト"""
        bqm = dimod.BinaryQuadraticModel({'q2': -1}, {('q0', 'q1'): 1}, vartype='BINARY')
        qudata = QuDataInput().from_dimod_bqm(bqm)
        expected = {('q0', 'q1'): 1, ('q2', 'q2'): -1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_dimod_bqm_invalid(self):
        """from_dimod_bqmメソッドで無効な型のデータを渡した場合のテスト"""
        qudata = QuDataInput()
        with self.assertRaises(TypeError):
            qudata.from_dimod_bqm("invalid")  # 無効な型でTypeErrorが発生するか確認

    def test_from_sympy(self):
        """from_sympyメソッドのテスト"""
        q0_sympy = Symbol('q0')
        q1_sympy = Symbol('q1')
        q2_sympy = Symbol('q2')
        prob_sympy = q0_sympy * q1_sympy - q2_sympy ** 2
        qudata = QuDataInput().from_sympy(prob_sympy)
        expected = {('q0', 'q1'): 1, ('q2', 'q2'): -1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_sympy_invalid(self):
        """from_sympyメソッドで無効な型のデータを渡した場合のテスト"""
        qudata = QuDataInput()
        with self.assertRaises(TypeError):
            qudata.from_sympy("invalid")  # 無効な型でTypeErrorが発生するか確認

##############################
### pulp
##############################
# 変数の定義
q0 = pulp.LpVariable('q0', lowBound=0, upBound=1, cat='Binary')
q1 = pulp.LpVariable('q1', lowBound=0, upBound=1, cat='Binary')

# 問題の定義 (2q0-q1)
problem = pulp.LpProblem('QUBO', pulp.LpMinimize)
problem += 2 * q0 - q1

##############################
### amplify
##############################
gen = VariableGenerator()
q = gen.array("Binary", shape=(3))
objective = q[0] * q[1] - q[2]

##############################
### array
##############################
array = np.array([
        [1, 1, 0],
        [0, 2, 0],
        [0, 0, -1],
    ])

##############################
### networkx
##############################
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (0, 2)])
# nx.draw_networkx(G)
# plt.show()

##############################
### pandas
##############################
df = pd.DataFrame(array,
                  columns=['q0', 'q1', 'q2'],
                  index=['q0', 'q1', 'q2'])

##############################
### dimod (bqm)
##############################
bqm = dimod.BinaryQuadraticModel({'q2': -1}, {('q0', 'q1'): 1}, vartype='BINARY')

##############################
### sympy
##############################
q0_sympy = Symbol('q0')
q1_sympy = Symbol('q1')
q2_sympy = Symbol('q2')
prob_sympy = q0_sympy * q1_sympy - q2_sympy ** 2
# print(prob.free_symbols)

# terms = prob.as_ordered_terms()
# print(terms, type(terms[0]), terms[1].free_symbols)

if __name__ == '__main__':
    unittest.main()

    # # dict
    # qd1 = QuDataInput({('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0})
    # print(f"dict={qd1.prob}")

    # # pulp
    # qd2 = QuDataInput()
    # qd2.from_pulp(problem)
    # print(f"pulp={qd2.prob}")

    # # amplify
    # qd3 = QuDataInput()
    # qd3.from_amplify(objective)
    # print(f"amplify={qd3.prob}")

    # # array
    # qd4 = QuDataInput()
    # qd4.from_array(array)
    # print(f"array={qd4.prob}")

    # # csv
    # qd5 = QuDataInput()
    # qd5.from_csv(path='./data/qudata.csv')
    # print(f"csv={qd5.prob}")

    # # json
    # qd6 = QuDataInput()
    # qd6.from_json(path='./data/qudata.json')
    # print(f"json={qd6.prob}")

    # # networkx
    # qd7 = QuDataInput()
    # qd7.from_networkx(G)
    # print(f"networkx={qd7.prob}")

    # # pandas
    # qd8 = QuDataInput()
    # qd8.from_pandas(df)
    # print(f"pandas={qd8.prob}")

    # # dimod (bqm)
    # qd9 = QuDataInput()
    # qd9.from_dimod_bqm(bqm)
    # print(f"dimod-bqm={qd9.prob}")

    # # sympy
    # qd10 = QuDataInput()
    # qd10.from_sympy(prob_sympy)
    # print(f"sympy={qd10.prob}")

    # # to_pulp
    # print(qd2.to_pulp())

    # # to_amplify
    # print(qd3.to_amplify())

    # # to_array
    # print(qd4.to_array())

    # # to_csv
    # qd5.to_csv(name="to-csv")

    # # to_json
    # qd6.to_json(name="to-json")

    # # to_networkx
    # toG = qd7.to_networkx()
    # # nx.draw_networkx(toG)
    # # plt.show()

    # # to_pandas
    # print(qd8.to_pandas())

    # # to_dimod-bqm
    # print(qd9.to_dimod_bqm())

    # # to_sympy
    # print(qd10.to_sympy())

    # # add
    # print(f"add={(qd1 + qd2).prob}")

    # # sub
    # print(f"sub={(qd1 - qd2).prob}")

    # # mul
    # print(f"mul={(qd1 * qd2).prob}")

    # # pow
    # print(f"pow={(qd1 ** 2).prob}")