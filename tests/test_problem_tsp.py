"""TSP Problem と複数 Algorithm。"""

from __future__ import annotations

import unittest

from qudas.problem import TSPProblem, MaxCutProblem, KnapsackProblem
from qudas.api import solve, solve_many, catalog


class TestTSP(unittest.TestCase):
    def test_brute_force_triangle(self):
        # 都市 0-1-2 の直角三角形
        dist = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
        p = TSPProblem(distances=dist)
        out = p.classical_brute_force()
        self.assertTrue(out["feasible"])
        self.assertAlmostEqual(out["objective"], 3.0)
        self.assertEqual(len(out["solution"]["tour"]), 3)

    def test_from_cities(self):
        p = TSPProblem.from_cities([(0, 0), (1, 0), (0, 1)])
        out = p.classical_brute_force()
        self.assertGreater(out["objective"], 0)

    def test_constraint_feasibility(self):
        p = TSPProblem(distances=[[0, 2], [2, 0]])
        bad = {v: 0 for v in p.variables}
        cons = p.evaluate_constraints(bad)
        self.assertFalse(cons["feasible"])

    def test_solve_api(self):
        p = TSPProblem(distances=[[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        res = solve(p, "classical.exact")
        self.assertEqual(res.status, "ok")
        self.assertTrue(res.feasibility)
        self.assertAlmostEqual(res.problem_objective, p.evaluate(res.solution))

    def test_annealing_dimod_small(self):
        p = TSPProblem(distances=[[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        res = solve(p, "annealing.dimod_exact")
        self.assertEqual(res.status, "ok")
        # ExactSolver は実行可能解を返すはず
        self.assertTrue(res.feasibility)
        self.assertAlmostEqual(res.problem_objective, 3.0)

    def test_catalog_includes_tsp(self):
        c = catalog()
        ids = [
            p["problem_id"] if isinstance(p, dict) else p for p in c["problems"]
        ]
        self.assertIn("tsp", ids)

    def test_independent_reevaluation_maxcut_knapsack_tsp(self):
        mc = MaxCutProblem.from_edge_list([(0, 1), (1, 2), (0, 2)])
        kn = KnapsackProblem(values=[2, 3, 2], weights=[1, 2, 1], capacity=3)
        tsp = TSPProblem(distances=[[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        for problem, algs in (
            (mc, ["classical.exact", "annealing.dimod_exact"]),
            (kn, ["classical.exact"]),
            (tsp, ["classical.exact"]),
        ):
            results = solve_many(problem, algorithms=algs)
            for r in results:
                self.assertEqual(r.status, "ok", r.message)
                # 目的関数の独立検算
                self.assertAlmostEqual(r.problem_objective, problem.evaluate(r.solution))


if __name__ == "__main__":
    unittest.main()
