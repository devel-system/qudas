"""QuDataOutput（最適化結果の取り込み・変換）のテスト。"""

from __future__ import annotations

import os
import unittest
from datetime import timedelta

import dimod
import numpy as np
from amplify import FixstarsClient, Model, VariableGenerator, solve
from pulp import LpMinimize, LpProblem, LpVariable
from scipy.optimize import Bounds, minimize
from sympy import lambdify, symbols

from qudas.qudata import QuDataOutput


class TestQuDataOutput(unittest.TestCase):
    def test_from_pulp(self):
        prob = LpProblem("Test Problem", LpMinimize)
        x = LpVariable('x', lowBound=0, upBound=1, cat='Binary')
        y = LpVariable('y', lowBound=0, upBound=1, cat='Binary')
        prob += 2 * x - y
        prob.solve()

        qdo = QuDataOutput().from_pulp(prob)
        expected_result = {'variables': {'x': 0, 'y': 1}, 'objective': -1}
        self.assertEqual(qdo.result, expected_result)
        self.assertEqual(qdo.result_type, 'pulp')

    def test_from_amplify(self):
        token = os.getenv("AMPLIFY_TOKEN")
        if not token:
            self.skipTest("環境変数 AMPLIFY_TOKEN が未設定のためスキップ")

        gen = VariableGenerator()
        q = gen.array("Binary", shape=(3))
        objective = 2 * q[0] - q[1] - q[2]

        client = FixstarsClient()
        client.token = token
        client.parameters.timeout = timedelta(milliseconds=100)

        amplify_result = solve(Model(objective), client)
        qdo = QuDataOutput().from_amplify(amplify_result)

        expected_result = {
            'variables': {'q_0': 0.0, 'q_1': 1.0, 'q_2': 1.0},
            'objective': -2,
        }
        self.assertEqual(qdo.result, expected_result)
        self.assertEqual(qdo.result_type, 'amplify')

    def test_from_dimod(self):
        qubo = {('q0', 'q0'): 2, ('q1', 'q1'): -1, ('q2', 'q2'): -1}
        sampleset = dimod.ExactSolver().sample_qubo(qubo)
        qdo = QuDataOutput().from_dimod(sampleset)
        expected_result = {
            'variables': {'q0': 0.0, 'q1': 1.0, 'q2': 1.0},
            'objective': -2,
        }
        self.assertEqual(qdo.result, expected_result)
        self.assertEqual(qdo.result_type, 'dimod')

    def test_from_scipy(self):
        q0, q1, q2 = symbols('q0 q1 q2')
        objective_function = 2 * q0 - q1 - q2
        f = lambdify([q0, q1, q2], objective_function, 'numpy')
        q = [0.5, 0.5, 0.5]
        bounds = Bounds([0, 0, 0], [1, 1, 1])
        res = minimize(lambda qv: f(qv[0], qv[1], qv[2]), q, method='SLSQP', bounds=bounds)

        qdo = QuDataOutput().from_scipy(res)
        expected_result = {
            'variables': {'q0': 0.0, 'q1': 1.0, 'q2': 1.0},
            'objective': -2,
        }
        self.assertEqual(qdo.result, expected_result)
        # 実装が result_type に 'sympy' を設定している（後方互換のため現状維持）
        self.assertEqual(qdo.result_type, 'sympy')

    def test_to_dimod(self):
        qdo = QuDataOutput(
            result={'variables': {'q0': 0.0, 'q1': 1.0, 'q2': 1.0}, 'objective': -2}
        )
        dimod_result = qdo.to_dimod()
        qubo = {('q0', 'q0'): 2, ('q1', 'q1'): -1, ('q2', 'q2'): -1}
        sampleset = dimod.ExactSolver().sample_qubo(qubo)
        self.assertEqual(dimod_result.first, sampleset.first)

    def test_to_scipy(self):
        qdo = QuDataOutput(
            result={'variables': {'q0': 0.0, 'q1': 1.0, 'q2': 1.0}, 'objective': -2}
        )
        scipy_result = qdo.to_scipy()

        q0, q1, q2 = symbols('q0 q1 q2')
        objective_function = 2 * q0 - q1 - q2
        f = lambdify([q0, q1, q2], objective_function, 'numpy')
        q = [0.5, 0.5, 0.5]
        bounds = Bounds([0, 0, 0], [1, 1, 1])
        res = minimize(lambda qv: f(qv[0], qv[1], qv[2]), q, method='SLSQP', bounds=bounds)

        np.testing.assert_array_equal(scipy_result.x, res.x)
        self.assertEqual(scipy_result.fun, res.fun)
        self.assertEqual(scipy_result.success, res.success)


if __name__ == '__main__':
    unittest.main()
