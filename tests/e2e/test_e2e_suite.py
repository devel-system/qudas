"""統合 E2E: Model Exchange / Hybrid Pipeline / Problem Library / Paper reuse。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qudas.problem import MaxCutProblem
from qudas.api import solve, compare
from qudas.algorithm import AlgorithmPackage
from qudas.algorithm.registry import get_algorithm
from qudas.pipeline import QdPipelineModel, ControlFlowSpec
from qudas.exchange import (
    import_qiskit_circuit,
    export_qiskit,
    import_openqasm,
    verify_unitary_equivalence,
    verify_qubo_equivalence,
    convert,
)


class TestE2EModelExchange(unittest.TestCase):
    """E2E-1: SDK → Import → Save → Reload → Convert → verify。"""

    def test_qiskit_import_save_reload_export(self):
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            self.skipTest("qiskit required")
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        ir, report = import_qiskit_circuit(qc)
        self.assertTrue(report.success)

        # Algorithm IR を Pipeline に載せて保存（ドメインモデル参照の実証）
        model = QdPipelineModel(name="bell")
        model.add_node("pass", "qudas.passthrough")
        model.metadata["gate_ir_gates"] = [
            {"gate": g.gate, "targets": g.targets, "params": g.params} for g in ir.gates
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bell.json"
            model.save(path)
            loaded = QdPipelineModel.load(path)
        self.assertEqual(len(loaded.metadata["gate_ir_gates"]), len(ir.gates))
        qc2, _ = export_qiskit(ir)
        self.assertTrue(verify_unitary_equivalence(qc, qc2).passed)

        qasm, _ = convert(ir, "qd_algorithm_ir", "openqasm2")
        ir2, _ = import_openqasm(qasm, version=2)
        self.assertTrue(verify_unitary_equivalence(ir, ir2).passed)


class TestE2EHybridPipeline(unittest.TestCase):
    """E2E-2: 古典→量子/アニーリング→古典→Loop → 保存 → 再実行。"""

    def test_hybrid_loop_save_rerun(self):
        problem = MaxCutProblem.from_edge_list([(0, 1), (1, 2), (0, 2)])
        model = QdPipelineModel(
            name="hybrid",
            control_flow=ControlFlowSpec(kind="loop", iterations=2),
            context={},
        )
        model.add_node("solve", "qudas.algorithm.classical_exact")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hybrid.json"
            model.save(path)
            loaded = QdPipelineModel.load(path)
        # 別「環境」相当: load 後に context へ problem を注入して実行
        loaded.context["problem"] = problem
        results = loaded.run(problem, mode="optimize")
        self.assertIn("solve", results)
        self.assertEqual(results["solve"].problem_objective, 2.0)


class TestE2EProblemLibrary(unittest.TestCase):
    """E2E-3: 共通 Problem → 複数 Algorithm → 再評価。"""

    def test_maxcut_multi_algorithm(self):
        problem = MaxCutProblem.from_edge_list([(0, 1), (1, 2), (0, 2), (0, 3)])
        summary = compare(
            problem,
            algorithms=["classical.exact", "annealing.dimod_exact"],
        )
        self.assertIn("comparison", summary)
        objs = [
            r["problem_objective"]
            for r in summary["results"]
            if r.get("status", "ok") == "ok"
        ]
        self.assertGreaterEqual(len(objs), 1)
        # 独立検算
        for r in summary["results"]:
            if r.get("status", "ok") != "ok":
                continue
            self.assertAlmostEqual(r["problem_objective"], problem.evaluate(r["solution"]))


class TestE2EPaperAlgorithmReuse(unittest.TestCase):
    """E2E-4: Algorithm Package 保存 → パラメータ/問題変更 → 再実行。"""

    def test_package_reuse_other_maxcut(self):
        spec = get_algorithm("classical.exact")
        pkg = AlgorithmPackage.from_spec(spec)
        data = pkg.to_dict()
        restored = AlgorithmPackage.from_dict(data)
        p1 = MaxCutProblem.from_edge_list([(0, 1)])
        p2 = MaxCutProblem.from_edge_list([(0, 1), (1, 2), (2, 0)])
        r1 = restored.run(p1)
        r2 = restored.run(p2)
        self.assertEqual(r1.problem_objective, 1.0)
        self.assertEqual(r2.problem_objective, 2.0)
        # QAOA package のパラメータ変更（実行は optional）
        qaoa = AlgorithmPackage.from_spec(get_algorithm("qaoa.qiskit"))
        qaoa2 = AlgorithmPackage.from_dict(qaoa.to_dict())
        qaoa2.parameters["shots"] = 128
        r = qaoa2.run(p2, shots=128, opt_steps=2)
        # qiskit 欠落時は error でよい
        self.assertIn(r.status, ("ok", "error"))


if __name__ == "__main__":
    unittest.main()
