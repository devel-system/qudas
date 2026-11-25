"""Pipeline Model のシリアライズ・検証・互換性。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qudas import QdPipeline, QdPipelineModel
from qudas.pipeline.registry import default_registry
from qudas.problem import MaxCutProblem


class TestPipelineModel(unittest.TestCase):
    def test_json_roundtrip(self):
        model = QdPipelineModel(name="t")
        model.add_node("a", "qudas.passthrough", parameters={"k": 1})
        model.add_node("b", "qudas.algorithm.classical_exact")
        model.add_edge("a", "b")
        text = model.to_json()
        restored = QdPipelineModel.from_json(text)
        self.assertEqual(restored.to_dict(), model.to_dict())

    def test_save_load(self):
        model = QdPipelineModel(name="t")
        model.add_node("solve", "qudas.algorithm.classical_exact")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "p.json"
            model.save(path)
            loaded = QdPipelineModel.load(path)
        self.assertEqual(loaded.nodes[0].type_id, "qudas.algorithm.classical_exact")

    def test_unregistered_type_errors(self):
        model = QdPipelineModel()
        model.add_node("x", "does.not.exist")
        errors = model.validate()
        self.assertTrue(any("Unregistered" in e for e in errors))
        with self.assertRaises(ValueError):
            model.materialize()

    def test_invalid_json_missing_payload(self):
        with self.assertRaises(ValueError):
            QdPipelineModel.from_json(json.dumps({"schema_version": "1.0.0"}))

    def test_parallel_control_flow_is_valid(self):
        model = QdPipelineModel()
        model.add_node("a", "qudas.passthrough")
        model.control_flow.kind = "parallel"
        errors = model.validate()
        self.assertFalse(any("schema-only" in e for e in errors))
        self.assertFalse(any("Unknown control_flow" in e for e in errors))
        self.assertTrue(model.needs_advanced_runtime())

    def test_materialize_and_run(self):
        problem = MaxCutProblem.from_edge_list([(0, 1), (1, 2), (2, 0)])
        model = QdPipelineModel(name="maxcut_classical")
        model.add_node("solve", "qudas.algorithm.classical_exact")
        results = model.run(problem, mode="optimize")
        self.assertIn("solve", results)
        self.assertEqual(results["solve"].problem_objective, 2.0)

    def test_with_parameters(self):
        model = QdPipelineModel()
        model.add_node("solve", "qudas.algorithm.qaoa_qiskit", parameters={"p": 1})
        updated = model.with_node_parameters("solve", shots=128)
        self.assertEqual(updated.nodes[0].parameters["shots"], 128)
        self.assertNotIn("shots", model.nodes[0].parameters)

    def test_existing_qdpipeline_still_works(self):
        """既存 QdPipeline API 互換（models は学習格納域のまま）。"""

        class S:
            def fit(self, X, y=None):
                return self

            def predict(self, X):
                return X

            def transform(self, X):
                return X

        pipe = QdPipeline([("s", S())])
        pipe.fit(1)
        self.assertIn("s", pipe.models)
        # Pipeline Model と混同しないこと
        self.assertFalse(hasattr(pipe, "nodes"))


if __name__ == "__main__":
    unittest.main()
