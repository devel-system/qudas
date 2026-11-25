"""共通 Result と QPLAT Contract。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qudas.api import solve, catalog
from qudas.problem import MaxCutProblem
from qudas.qplat import QplatContract
from qudas.result import QdExperimentResult
from qudas.algorithm import get_algorithm


class TestResultQplat(unittest.TestCase):
    def test_result_roundtrip(self):
        p = MaxCutProblem.from_edge_list([(0, 1)])
        r = solve(p, "classical.exact")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "r.json"
            r.save(path)
            loaded = QdExperimentResult.load(path)
        self.assertEqual(loaded.problem_objective, r.problem_objective)

    def test_qplat_contract(self):
        problems = QplatContract.list_problems()
        algorithms = QplatContract.list_algorithms()
        self.assertTrue(any(p["problem_id"] == "maxcut" for p in problems))
        self.assertTrue(any(a["algorithm_id"] == "qaoa.qiskit" for a in algorithms))
        schema = QplatContract.get_algorithm_schema("qaoa.qiskit")
        self.assertIn("parameters", schema)
        model = get_algorithm("classical.exact").build_pipeline()
        viz = QplatContract.pipeline_visualization(model)
        self.assertIn("nodes", viz)
        self.assertIn("edges", viz)

    def test_catalog(self):
        c = catalog()
        self.assertIn("problems", c)
        self.assertIn("algorithms", c)


if __name__ == "__main__":
    unittest.main()
