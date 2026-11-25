"""Algorithm Package の保存・再利用・Adapt。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qudas.algorithm import AlgorithmPackage, get_algorithm
from qudas.exchange import adapt_qaoa_to_problem, check_conversion
from qudas.problem import MaxCutProblem
from qudas.model import ConversionKind


class TestAlgorithmPackage(unittest.TestCase):
    def test_save_load_and_reuse(self):
        spec = get_algorithm("annealing.dimod_exact")
        pkg = AlgorithmPackage.from_spec(
            spec,
            paper={"arXiv": "none", "note": "built-in solver"},
            license="Apache-2.0",
        )
        p1 = MaxCutProblem.from_edge_list([(0, 1), (1, 2), (2, 0)])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pkg.json"
            pkg.save(path)
            loaded = AlgorithmPackage.load(path)
        r1 = loaded.run(p1)
        self.assertEqual(r1.problem_objective, 2.0)
        # 問題データを変更して同一 Algorithm を再利用
        p2 = MaxCutProblem.from_edge_list([(0, 1), (1, 2), (2, 3), (3, 0)])
        r2 = loaded.run(p2)
        self.assertEqual(r2.problem_objective, 4.0)

    def test_pipeline_model_in_package(self):
        spec = get_algorithm("classical.exact")
        pkg = AlgorithmPackage.from_spec(spec)
        self.assertIsNotNone(pkg.pipeline_model)
        self.assertEqual(pkg.pipeline_model.validate(), [])


class TestExchange(unittest.TestCase):
    def test_unsupported_gate_to_annealing(self):
        report = check_conversion("gate", "annealing")
        self.assertEqual(report.kind, ConversionKind.UNSUPPORTED)
        self.assertFalse(report.success)

    def test_qiskit_roundtrip(self):
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            self.skipTest("qiskit not installed")
        from qudas.exchange import import_qiskit_circuit, export_qiskit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        ir, report = import_qiskit_circuit(qc)
        self.assertTrue(report.success)
        qc2, report2 = export_qiskit(ir)
        self.assertTrue(report2.success)
        self.assertEqual(qc2.num_qubits, 2)

    def test_openqasm_import(self):
        try:
            import qiskit  # noqa: F401
        except ImportError:
            self.skipTest("qiskit not installed")
        from qudas.exchange import import_openqasm

        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[1];\nh q[0];\n"
        )
        ir, report = import_openqasm(qasm)
        self.assertTrue(report.success)
        self.assertGreaterEqual(len(ir.gates), 1)

    def test_adapt_qaoa(self):
        try:
            import qiskit_aer  # noqa: F401
        except ImportError:
            self.skipTest("qiskit-aer not installed")
        p = MaxCutProblem.from_edge_list([(0, 1), (1, 2), (2, 0)])
        result = adapt_qaoa_to_problem(
            p, parameters={"p": 1, "shots": 128, "opt_steps": 3, "seed": 1}
        )
        self.assertEqual(result.algorithm, "qaoa.qiskit")
        self.assertIn(result.status, ("ok", "error"))


if __name__ == "__main__":
    unittest.main()
