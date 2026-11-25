"""MaxCut / Knapsack 変換・厳密解・Solver 一致。"""

from __future__ import annotations

import unittest

from qudas.problem import MaxCutProblem, KnapsackProblem
from qudas.api import solve
from qudas.result import compare_results


class TestMaxCut(unittest.TestCase):
    def setUp(self):
        # 三角形: 最適カット = 2
        self.triangle = MaxCutProblem.from_edge_list([(0, 1), (1, 2), (2, 0)])

    def test_qubo_energy_matches_neg_cut(self):
        bf = self.triangle.classical_brute_force()
        sol = bf["solution"]
        cut = self.triangle.cut_value(sol)
        energy = self.triangle.evaluate_energy(sol)
        self.assertEqual(cut, 2.0)
        self.assertEqual(energy, -cut)

    def test_ising_roundtrip_via_dimod(self):
        ising = self.triangle.to_model("ising")
        self.assertEqual(ising["vartype"], "SPIN")
        self.assertIn("linear", ising)
        self.assertIn("quadratic", ising)

    def test_brute_force_enumeration(self):
        out = self.triangle.classical_brute_force()
        self.assertEqual(out["objective"], 2.0)

    def test_classical_and_annealing_agree(self):
        r1 = solve(self.triangle, "classical.exact")
        r2 = solve(self.triangle, "annealing.dimod_exact")
        self.assertEqual(r1.status, "ok")
        self.assertEqual(r2.status, "ok")
        self.assertEqual(r1.problem_objective, r2.problem_objective)
        # 最小化エネルギーは -cut
        self.assertAlmostEqual(r2.minimized_energy, -r2.problem_objective)

    def test_weighted_graph(self):
        p = MaxCutProblem.from_edge_list([(0, 1, 2.5), (1, 2, 1.0)])
        out = p.classical_brute_force()
        self.assertEqual(out["objective"], 3.5)

    def test_from_networkx(self):
        try:
            import networkx as nx
        except ImportError:
            self.skipTest("networkx not installed")
        g = nx.cycle_graph(4)
        p = MaxCutProblem.from_networkx(g)
        self.assertEqual(p.classical_brute_force()["objective"], 4.0)

    def test_compare_results(self):
        results = [
            solve(self.triangle, "classical.exact"),
            solve(self.triangle, "annealing.dimod_exact"),
        ]
        table = compare_results(results)
        self.assertEqual(table["count"], 2)
        self.assertIsNotNone(table["best"])


class TestKnapsack(unittest.TestCase):
    def test_reuse_binary_model(self):
        p = KnapsackProblem(values=[4, 3, 2], weights=[3, 2, 1], capacity=3)
        out = p.classical_brute_force()
        self.assertEqual(out["objective"], 5.0)  # items 1+2 -> value 3+2
        r = solve(p, "classical.exact")
        self.assertEqual(r.problem_objective, 5.0)
        # annealing も同じ最適値（ペナルティ付きだが厳密解器）
        r2 = solve(p, "annealing.dimod_exact")
        self.assertEqual(r2.status, "ok")
        self.assertEqual(r2.problem_objective, 5.0)
        self.assertTrue(r2.feasibility)


if __name__ == "__main__":
    unittest.main()
