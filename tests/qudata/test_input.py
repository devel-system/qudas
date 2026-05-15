"""QuDataInput（QUBO 入出力・各フォーマット変換）のテスト。"""

from __future__ import annotations

import unittest
from pathlib import Path

import dimod
import networkx as nx
import numpy as np
import pandas as pd
import pulp
from amplify import VariableGenerator
from sympy import Symbol

from qudas.annealing import QdAnnealingIR
from qudas.qudata import QuDataInput

_DATA = Path(__file__).resolve().parent.parent / "data"


def dicts_are_equal(dict1, dict2):
    """辞書のキーの順序を無視して等価性を比較する。"""
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


class TestQuDataInput(unittest.TestCase):
    def test_init_with_dict(self):
        prob = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        qudata = QuDataInput(prob)
        self.assertTrue(dicts_are_equal(qudata.prob, prob))

    def test_init_with_none(self):
        qudata = QuDataInput()
        self.assertEqual(qudata.prob, {})

    def test_init_with_invalid_type(self):
        with self.assertRaises(TypeError):
            QuDataInput(123)

    def test_add(self):
        prob1 = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        prob2 = {('q0', 'q0'): 2.0, ('q1', 'q1'): -1.0}
        qudata1 = QuDataInput(prob1)
        qudata2 = QuDataInput(prob2)
        result = qudata1 + qudata2
        expected = {
            ('q0', 'q1'): 1.0,
            ('q2', 'q2'): -1.0,
            ('q0', 'q0'): 2,
            ('q1', 'q1'): -1,
        }
        self.assertTrue(dicts_are_equal(result.prob, expected))

    def test_sub(self):
        prob1 = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        prob2 = {('q0', 'q0'): 2.0, ('q1', 'q1'): -1.0}
        qudata1 = QuDataInput(prob1)
        qudata2 = QuDataInput(prob2)
        result = qudata1 - qudata2
        expected = {
            ('q0', 'q1'): 1.0,
            ('q2', 'q2'): -1.0,
            ('q0', 'q0'): -2,
            ('q1', 'q1'): 1,
        }
        self.assertTrue(dicts_are_equal(result.prob, expected))

    def test_mul(self):
        prob1 = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        prob2 = {('q0', 'q0'): 2.0, ('q1', 'q1'): -1.0}
        qudata1 = QuDataInput(prob1)
        qudata2 = QuDataInput(prob2)
        result = qudata1 * qudata2
        expected = {('q0', 'q1'): 1.0, ('q0', 'q2'): -2.0, ('q1', 'q2'): 1.0}
        self.assertTrue(dicts_are_equal(result.prob, expected))

    def test_pow(self):
        prob = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        qudata = QuDataInput(prob)
        result = qudata**2
        expected = {('q0', 'q1'): 1.0, ('q0', 'q2', 'q1'): -2.0, ('q2', 'q2'): 1.0}
        self.assertTrue(dicts_are_equal(result.prob, expected))

    def test_pow_invalid_type(self):
        qudata = QuDataInput({('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0})
        with self.assertRaises(TypeError):
            qudata ** 'invalid'

    def test_from_pulp(self):
        q0 = pulp.LpVariable('q0', lowBound=0, upBound=1, cat='Binary')
        q1 = pulp.LpVariable('q1', lowBound=0, upBound=1, cat='Binary')
        problem = pulp.LpProblem('QUBO', pulp.LpMinimize)
        problem += 2 * q0 - q1
        qudata = QuDataInput().from_pulp(problem)
        expected = {('q0', 'q0'): 2, ('q1', 'q1'): -1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_pulp_invalid_type(self):
        qudata = QuDataInput()
        with self.assertRaises(TypeError):
            qudata.from_pulp("invalid")

    def test_from_amplify(self):
        q = VariableGenerator().array("Binary", shape=(3))
        objective = q[0] * q[1] - q[2]
        qudata = QuDataInput().from_amplify(objective)
        expected = {('q_0', 'q_1'): 1.0, ('q_2', 'q_2'): -1.0}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_amplify_invalid_type(self):
        qudata = QuDataInput()
        with self.assertRaises(TypeError):
            qudata.from_amplify("invalid")

    def test_from_pyqubo(self):
        try:
            from pyqubo import Binary
        except ImportError:
            self.skipTest("pyqubo がインストールされていないためスキップ")

        q0, q1 = Binary("q0"), Binary("q1")
        prob = (q0 + q1) ** 2
        qudata = QuDataInput().from_pyqubo(prob)
        expected = {('q0', 'q0'): 1.0, ('q0', 'q1'): 2.0, ('q1', 'q1'): 1.0}
        self.assertEqual(qudata.prob, expected)

    def test_from_pyqubo_invalid_type(self):
        qudata = QuDataInput()
        with self.assertRaises(TypeError):
            qudata.from_pyqubo("invalid")

    def test_from_array(self):
        prob = np.array(
            [
                [1, 1, 0],
                [0, 2, 0],
                [0, 0, -1],
            ]
        )
        qudata = QuDataInput().from_array(prob)
        expected = {
            ('q_0', 'q_0'): 1,
            ('q_0', 'q_1'): 1,
            ('q_1', 'q_1'): 2,
            ('q_2', 'q_2'): -1,
        }
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_array_invalid_type(self):
        qudata = QuDataInput()
        with self.assertRaises(TypeError):
            qudata.from_array("invalid")

    def test_from_csv(self):
        qudata = QuDataInput().from_csv(str(_DATA / "qudata.csv"))
        expected = {
            ('q_0', 'q_0'): 1.0,
            ('q_0', 'q_2'): 2.0,
            ('q_1', 'q_1'): -1.0,
            ('q_2', 'q_1'): 2.0,
            ('q_2', 'q_2'): 2.0,
        }
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_csv_invalid(self):
        qudata = QuDataInput()
        with self.assertRaises(ValueError, msg="読み取りエラー"):
            qudata.from_csv(str(_DATA / "invalid_data.csv"))

    def test_from_json(self):
        qudata = QuDataInput().from_json(str(_DATA / "qudata.json"))
        expected = {
            ('q0', 'q0'): 1.0,
            ('q0', 'q1'): 1.0,
            ('q1', 'q1'): -1.0,
            ('q2', 'q2'): 2.0,
        }
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_json_invalid(self):
        qudata = QuDataInput()
        with self.assertRaises(ValueError, msg="読み取りエラー"):
            qudata.from_json(str(_DATA / "invalid_data.json"))

    def test_from_networkx(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (0, 2)])
        qudata = QuDataInput().from_networkx(G)
        expected = {('q_0', 'q_1'): 1, ('q_1', 'q_2'): 1, ('q_0', 'q_2'): 1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_networkx_invalid(self):
        qudata = QuDataInput()
        with self.assertRaises(TypeError):
            qudata.from_networkx("invalid")

    def test_from_pandas(self):
        array = np.array(
            [
                [1, 1, 0],
                [0, 2, 0],
                [0, 0, -1],
            ]
        )
        df = pd.DataFrame(array, columns=['q0', 'q1', 'q2'], index=['q0', 'q1', 'q2'])
        qudata = QuDataInput().from_pandas(df)
        expected = {('q0', 'q0'): 1, ('q0', 'q1'): 1, ('q1', 'q1'): 2, ('q2', 'q2'): -1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_pandas_invalid(self):
        qudata = QuDataInput()
        with self.assertRaises(TypeError):
            qudata.from_pandas("invalid")

    def test_from_dimod_bqm(self):
        bqm = dimod.BinaryQuadraticModel(
            {'q2': -1}, {('q0', 'q1'): 1}, vartype='BINARY'
        )
        qudata = QuDataInput().from_dimod_bqm(bqm)
        expected = {('q0', 'q1'): 1, ('q2', 'q2'): -1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_dimod_bqm_invalid(self):
        qudata = QuDataInput()
        with self.assertRaises(TypeError):
            qudata.from_dimod_bqm("invalid")

    def test_from_sympy(self):
        q0_sympy = Symbol('q0')
        q1_sympy = Symbol('q1')
        q2_sympy = Symbol('q2')
        prob_sympy = q0_sympy * q1_sympy - q2_sympy**2
        qudata = QuDataInput().from_sympy(prob_sympy)
        expected = {('q0', 'q1'): 1, ('q2', 'q2'): -1}
        self.assertTrue(dicts_are_equal(qudata.prob, expected))

    def test_from_sympy_invalid(self):
        qudata = QuDataInput()
        with self.assertRaises(TypeError):
            qudata.from_sympy("invalid")

    def test_to_pulp(self):
        qudata = QuDataInput({('q0', 'q0'): 2, ('q1', 'q1'): -1})
        prob = qudata.to_pulp()
        self.assertIsInstance(prob, pulp.LpProblem)

        q0 = pulp.LpVariable('q0', lowBound=0, upBound=1, cat='Binary')
        q1 = pulp.LpVariable('q1', lowBound=0, upBound=1, cat='Binary')
        problem = pulp.LpProblem('QUBO', pulp.LpMinimize)
        problem += 2 * q0 - q1

        self.assertEqual(str(prob.objective), str(problem.objective))
        self.assertEqual(
            [v.name for v in prob.variables()],
            [v.name for v in problem.variables()],
        )

    def test_to_amplify(self):
        from amplify import Poly

        qudata = QuDataInput({('q_0', 'q_1'): 1.0, ('q_2', 'q_2'): -1.0})
        prob = qudata.to_amplify()
        self.assertIsInstance(prob, Poly)

        q = VariableGenerator().array("Binary", shape=(3))
        objective = q[0] * q[1] - q[2]
        # Poly の str 表記は項の順序に依存するため、QUBO 辞書へ正規化して比較する
        qubo_prob = QdAnnealingIR().from_amplify(prob).qubo
        qubo_ref = QdAnnealingIR().from_amplify(objective).qubo
        self.assertTrue(dicts_are_equal(qubo_prob, qubo_ref))

    def test_to_pyqubo(self):
        try:
            from pyqubo import Binary
            from pyqubo.utils.asserts import assert_qubo_equal
        except ImportError:
            self.skipTest("pyqubo がインストールされていないためスキップ")

        qudata = QuDataInput({('q0', 'q0'): 1.0, ('q0', 'q1'): 2.0, ('q1', 'q1'): 1.0})
        prob = qudata.to_pyqubo()
        q0, q1 = Binary("q0"), Binary("q1")
        objective = (q0 + q1) ** 2
        qubo1, _ = prob.compile().to_qubo()
        qubo2, _ = objective.compile().to_qubo()
        assert_qubo_equal(qubo1, qubo2)

    def test_to_array(self):
        qudata = QuDataInput(
            {
                ('q_0', 'q_0'): 1,
                ('q_0', 'q_1'): 1,
                ('q_1', 'q_1'): 2,
                ('q_2', 'q_2'): -1,
            }
        )
        prob = qudata.to_array()
        array = np.array(
            [
                [1, 1, 0],
                [0, 2, 0],
                [0, 0, -1],
            ]
        )
        np.testing.assert_array_equal(prob, array)

    def test_to_csv(self):
        import os

        filename = "test_qudata_write"
        qudata = QuDataInput(
            {
                ('q0', 'q0'): 1,
                ('q0', 'q2'): 2,
                ('q1', 'q1'): -1,
                ('q2', 'q1'): 2,
                ('q2', 'q2'): 2,
            }
        )
        qudata.to_csv(name=filename)
        try:
            self.assertTrue(os.path.exists(f"{filename}.csv"))
        finally:
            path = f"{filename}.csv"
            if os.path.exists(path):
                os.remove(path)

    def test_to_json(self):
        import os

        filename = "test_qudata_write_json"
        qudata = QuDataInput(
            {('q0', 'q0'): 1, ('q0', 'q1'): 1, ('q1', 'q1'): -1, ('q2', 'q2'): 2}
        )
        qudata.to_json(name=filename)
        try:
            self.assertTrue(os.path.exists(f"{filename}.json"))
        finally:
            path = f"{filename}.json"
            if os.path.exists(path):
                os.remove(path)

    def test_to_networkx(self):
        qudata = QuDataInput({('q_0', 'q_1'): 1, ('q_1', 'q_2'): 1, ('q_0', 'q_2'): 1})
        G = qudata.to_networkx()
        self.assertIsInstance(G, nx.Graph)
        H = nx.Graph()
        H.add_edges_from([(0, 1), (1, 2), (0, 2)])
        self.assertEqual(set(G.nodes()), set(H.nodes()))
        # 無向グラフでは (u,v) と (v,u) が同一辺として表現されうる
        edges_g = {frozenset(e) for e in G.edges()}
        edges_h = {frozenset(e) for e in H.edges()}
        self.assertEqual(edges_g, edges_h)

    def test_to_pandas(self):
        qudata = QuDataInput(
            {('q0', 'q0'): 1, ('q0', 'q1'): 1, ('q1', 'q1'): 2, ('q2', 'q2'): -1}
        )
        df = qudata.to_pandas()
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
        qudata = QuDataInput({('q0', 'q1'): 1, ('q2', 'q2'): -1})
        bqm = qudata.to_dimod_bqm()
        self.assertIsInstance(bqm, dimod.BinaryQuadraticModel)
        expected_bqm = dimod.BinaryQuadraticModel(
            {'q2': -1}, {('q0', 'q1'): 1}, vartype='BINARY'
        )
        self.assertEqual(bqm, expected_bqm)

    def test_to_sympy(self):
        qudata = QuDataInput({('q0', 'q1'): 1, ('q2', 'q2'): -1})
        expr = qudata.to_sympy()
        q0_sympy = Symbol('q0')
        q1_sympy = Symbol('q1')
        q2_sympy = Symbol('q2')
        prob_sympy = q0_sympy * q1_sympy - q2_sympy
        self.assertEqual(expr, prob_sympy)


if __name__ == '__main__':
    unittest.main()
