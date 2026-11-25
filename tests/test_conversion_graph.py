"""Conversion Graph / 意味保存 / Capability。"""

from __future__ import annotations

import unittest

from qudas.model.graph import ConversionGraph, weakest_kind
from qudas.model.schema import ConversionKind
from qudas.model.conversion import ConversionError
from qudas.exchange import (
    check_conversion,
    convert,
    ensure_graph_initialized,
    list_capabilities,
    list_openqasm3_features,
    import_openqasm,
    import_qiskit_circuit,
    export_qiskit,
    verify_qubo_equivalence,
    verify_qubo_energy,
    verify_unitary_equivalence,
    verify_measurement_probabilities,
    get_qir_contract,
)


class TestConversionGraph(unittest.TestCase):
    def setUp(self):
        ensure_graph_initialized()

    def test_capabilities_nonempty(self):
        caps = list_capabilities()
        self.assertGreater(len(caps), 5)

    def test_direct_path_qiskit_ir(self):
        report = check_conversion("qiskit", "qd_algorithm_ir")
        self.assertTrue(report.success)
        self.assertEqual(report.kind, ConversionKind.LOSSLESS)

    def test_multihop_openqasm2_to_ir(self):
        report = check_conversion("openqasm2", "qd_algorithm_ir")
        self.assertTrue(report.success)

    def test_gate_annealing_unsupported(self):
        report = check_conversion("gate", "annealing")
        self.assertFalse(report.success)
        self.assertEqual(report.kind, ConversionKind.UNSUPPORTED)

    def test_weakest_kind(self):
        self.assertEqual(
            weakest_kind([ConversionKind.LOSSLESS, ConversionKind.APPROXIMATE]),
            ConversionKind.APPROXIMATE,
        )
        self.assertEqual(
            weakest_kind([ConversionKind.LOSSLESS, ConversionKind.PROBLEM_LEVEL]),
            ConversionKind.PROBLEM_LEVEL,
        )

    def test_custom_graph_path_selection(self):
        g = ConversionGraph()
        g.register("a", "b", ConversionKind.LOSSLESS, converter=lambda x: x + 1)
        g.register("b", "c", ConversionKind.APPROXIMATE, converter=lambda x: x * 2)
        g.register("a", "c", ConversionKind.PROBLEM_LEVEL, converter=lambda x: x + 100)
        paths = g.find_paths("a", "c")
        self.assertGreaterEqual(len(paths), 1)
        # shortest prefer: direct
        out, report = g.convert(1, "a", "c", prefer="shortest")
        self.assertEqual(out, 101)
        self.assertEqual(report.kind, ConversionKind.PROBLEM_LEVEL)
        out2, report2 = g.convert(1, "a", "c", prefer="lossless")
        # no full lossless path; falls back to first
        self.assertIn(report2.kind, (ConversionKind.PROBLEM_LEVEL, ConversionKind.APPROXIMATE))

    def test_qubo_ising_roundtrip_energy(self):
        qubo = {("x0", "x0"): 1.0, ("x1", "x1"): -1.0, ("x0", "x1"): 0.5}
        ising, report = convert(qubo, "qubo", "ising")
        self.assertTrue(report.success)
        back, report2 = convert(ising, "ising", "qubo")
        self.assertTrue(report2.success)
        # エネルギー一致（同一 assignment）
        assign = {"x0": 1, "x1": 0}
        e1 = verify_qubo_energy(qubo, assign, expected_energy=1.0)
        self.assertTrue(e1.passed)
        check = verify_qubo_equivalence(qubo, back)
        self.assertTrue(check.passed, check.message)

    def test_qiskit_roundtrip_unitary(self):
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            self.skipTest("qiskit required")
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        ir, report = import_qiskit_circuit(qc)
        self.assertTrue(report.success)
        qc2, report2 = export_qiskit(ir)
        self.assertTrue(report2.success)
        ur = verify_unitary_equivalence(qc, qc2)
        self.assertTrue(ur.passed, ur.message)

    def test_openqasm2_import(self):
        qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
"""
        ir, report = import_openqasm(qasm, version=2)
        self.assertTrue(report.success)
        self.assertGreaterEqual(len(ir.gates), 2)

    def test_openqasm3_basic_or_skip(self):
        features = list_openqasm3_features()
        self.assertIn("basic_gates", features)
        qasm = """
OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""
        try:
            ir, report = import_openqasm(qasm, version=3)
        except ConversionError as exc:
            # 環境により qasm3 が失敗しうる
            self.assertIsNotNone(exc.report)
            return
        self.assertTrue(report.success)
        qc, _ = export_qiskit(ir)
        ur = verify_unitary_equivalence(ir, qc)
        self.assertTrue(ur.passed or ur.details.get("skipped"))

    def test_openqasm3_rejects_classical_control(self):
        qasm = """
OPENQASM 3;
qubit q;
bit c;
c = measure q;
if (c) { x q; }
"""
        with self.assertRaises(ConversionError):
            import_openqasm(qasm, version=3)

    def test_measure_not_unitary_only(self):
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            self.skipTest("qiskit required")
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        ir, _ = import_qiskit_circuit(qc)
        ur = verify_unitary_equivalence(qc, ir)
        self.assertFalse(ur.passed)
        self.assertIn("measurements", ur.message)
        pr = verify_measurement_probabilities(qc, ir)
        self.assertTrue(pr.passed, pr.message)

    def test_qir_contract(self):
        c = get_qir_contract()
        self.assertEqual(c["status"], "not_implemented")
        self.assertFalse(c["import_supported"])


if __name__ == "__main__":
    unittest.main()
