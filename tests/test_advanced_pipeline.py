"""Advanced Pipeline Runtime: Branch / Join / Sub / Parallel / Loop。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qudas.pipeline import (
    QdPipelineModel,
    ControlFlowSpec,
    AdvancedPipelineRuntime,
    default_registry,
    register_builtin,
)
from qudas.pipeline.base import QdBaseStep, OptimizerStep
from qudas.exchange import verify_pipeline_types


@register_builtin("qudas.test.add")
class AddStep(OptimizerStep):
    def __init__(self, delta: int = 1, **kwargs):
        super().__init__()
        self.delta = int(delta)
        self.params = dict(kwargs)

    def optimize(self, X=None, y=None):
        return int(X) + self.delta


@register_builtin("qudas.test.mul")
class MulStep(OptimizerStep):
    def __init__(self, factor: int = 2, **kwargs):
        super().__init__()
        self.factor = int(factor)

    def optimize(self, X=None, y=None):
        return int(X) * self.factor


@register_builtin("qudas.test.flag")
class FlagStep(OptimizerStep):
    """入力をそのまま返し、context に flag を書く用途は metadata 側。"""

    def __init__(self, **kwargs):
        super().__init__()

    def optimize(self, X=None, y=None):
        return X


class TestAdvancedPipeline(unittest.TestCase):
    def test_branch_join_true_path(self):
        model = QdPipelineModel(
            name="branch",
            control_flow=ControlFlowSpec(
                kind="branch",
                details={
                    "condition": {"path": "value", "op": "gt", "value": 0},
                    "true_nodes": ["t"],
                    "false_nodes": ["f"],
                    "join": "j",
                },
            ),
        )
        model.add_node("t", "qudas.test.add", parameters={"delta": 10})
        model.add_node("f", "qudas.test.add", parameters={"delta": -10})
        model.add_node(
            "j",
            "qudas.passthrough",
            metadata={"role": "join", "join": "last"},
        )
        out = model.run(5, mode="optimize")
        self.assertEqual(out, 15)

    def test_branch_false_path(self):
        model = QdPipelineModel(
            name="branch",
            control_flow=ControlFlowSpec(
                kind="branch",
                details={
                    "condition": {"path": "value", "op": "gt", "value": 0},
                    "true_nodes": ["t"],
                    "false_nodes": ["f"],
                },
            ),
        )
        model.add_node("t", "qudas.test.add", parameters={"delta": 10})
        model.add_node("f", "qudas.test.add", parameters={"delta": -3})
        out = model.run(-2, mode="optimize")
        self.assertEqual(out, -5)

    def test_graph_branch_labels(self):
        model = QdPipelineModel(
            name="g",
            control_flow=ControlFlowSpec(kind="graph"),
        )
        model.add_node(
            "br",
            "qudas.passthrough",
            metadata={
                "role": "branch",
                "condition": {"path": "context.flag", "equals": True},
            },
        )
        model.add_node("yes", "qudas.test.mul", parameters={"factor": 3})
        model.add_node("no", "qudas.test.mul", parameters={"factor": 5})
        model.add_edge("br", "yes", label="true")
        model.add_edge("br", "no", label="false")
        model.context["flag"] = True
        out = model.run(2, mode="optimize")
        self.assertEqual(out, 6)

    def test_subpipeline_reuse(self):
        inner = QdPipelineModel(name="inner")
        inner.add_node("a", "qudas.test.add", parameters={"delta": 1})
        inner.add_node("b", "qudas.test.mul", parameters={"factor": 4})
        inner.add_edge("a", "b")

        outer = QdPipelineModel(name="outer", control_flow=ControlFlowSpec(kind="graph"))
        outer.add_node(
            "sub",
            "qudas.subpipeline",
            parameters={"model": inner.to_dict()},
        )
        outer.add_node("c", "qudas.test.add", parameters={"delta": 1})
        outer.add_edge("sub", "c")
        # (2+1)*4 + 1 = 13
        out = outer.run(2, mode="optimize")
        self.assertEqual(out, 13)

    def test_parallel_branches_logical(self):
        model = QdPipelineModel(
            name="par",
            control_flow=ControlFlowSpec(
                kind="parallel",
                details={"branches": [["a"], ["b"]], "join": "j"},
            ),
        )
        model.add_node("a", "qudas.test.add", parameters={"delta": 1})
        model.add_node("b", "qudas.test.mul", parameters={"factor": 10})
        model.add_node(
            "j",
            "qudas.passthrough",
            metadata={"role": "join", "join": "list"},
        )
        out = model.run(3, mode="optimize")
        self.assertEqual(out, [4, 30])

    def test_loop_with_max_iterations(self):
        model = QdPipelineModel(
            name="loop",
            control_flow=ControlFlowSpec(
                kind="loop",
                iterations=100,
                details={
                    "max_iterations": 3,
                    "terminate_when": {"path": "value", "op": "ge", "value": 1000},
                },
            ),
        )
        model.add_node("a", "qudas.test.add", parameters={"delta": 1})
        out = model.run(0, mode="optimize")
        self.assertEqual(out, 3)

    def test_json_roundtrip_and_rerun(self):
        model = QdPipelineModel(
            name="port",
            control_flow=ControlFlowSpec(
                kind="branch",
                details={
                    "condition": {"path": "value", "op": "gt", "value": 0},
                    "true_nodes": ["t"],
                    "false_nodes": ["f"],
                },
            ),
            dependencies=["qudas"],
            execution_requirements={"python": ">=3.8"},
        )
        model.add_node("t", "qudas.test.add", parameters={"delta": 2}, metadata={"op_version": "1"})
        model.add_node("f", "qudas.test.add", parameters={"delta": 0})
        check = verify_pipeline_types(model)
        self.assertTrue(check.passed, check.message)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.json"
            model.save(path)
            loaded = QdPipelineModel.load(path)
        self.assertEqual(loaded.dependencies, ["qudas"])
        self.assertEqual(loaded.run(1, mode="optimize"), 3)

    def test_secret_rejected(self):
        model = QdPipelineModel(parameters={"api_key": "x"})
        model.add_node("a", "qudas.passthrough")
        errors = model.validate()
        self.assertTrue(any("secret" in e or "api_key" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
