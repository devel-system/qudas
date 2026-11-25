"""Gate Domain の Import / Port。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from qudas.model.conversion import ConversionError, ConversionReport
from qudas.model.schema import ConversionKind
from .capabilities import check_conversion


def import_qiskit_circuit(circuit: Any):
    """Qiskit QuantumCircuit → QdAlgorithmIR。"""
    from qudas.gate import QdAlgorithmIR

    report = check_conversion("qiskit", "qd_algorithm_ir")
    if not report.success:
        raise ConversionError(report.message, report)
    # from_qasm は QuantumCircuit も受け付ける
    ir = QdAlgorithmIR.from_qasm(circuit)
    return ir, report


def import_openqasm(qasm: str):
    """OpenQASM 2 文字列/ファイル → QdAlgorithmIR（Qiskit 経由）。"""
    from qudas.gate import QdAlgorithmIR

    report = check_conversion("openqasm2", "qd_algorithm_ir")
    if not report.success:
        raise ConversionError(report.message, report)
    try:
        ir = QdAlgorithmIR.from_qasm(qasm)
    except Exception as exc:
        # OpenQASM3 の可能性
        q3 = check_conversion("openqasm3", "qd_algorithm_ir")
        raise ConversionError(
            f"Failed to parse as OpenQASM2 via Qiskit: {exc}. "
            f"OpenQASM3 built-in support: available={q3.success} ({q3.message})",
            q3,
        ) from exc
    return ir, report


def export_qiskit(algorithm_ir: Any):
    """QdAlgorithmIR → Qiskit QuantumCircuit。"""
    report = check_conversion("qd_algorithm_ir", "qiskit")
    if not report.success:
        raise ConversionError(report.message, report)
    try:
        qc = algorithm_ir.to_qiskit()
    except ValueError as exc:
        # 未対応ゲート
        approx = ConversionReport(
            kind=ConversionKind.UNSUPPORTED,
            source="qd_algorithm_ir",
            target="qiskit",
            success=False,
            message=str(exc),
            constraints=["supported_gates: x,h,cx,rz,rx,ry,measure"],
        )
        raise ConversionError(str(exc), approx) from exc
    return qc, report


def port_gate(source_obj: Any, source_format: str, target_format: str):
    """Gate 系の Port。未対応は明示失敗。"""
    report = check_conversion(source_format, target_format)
    if not report.success:
        raise ConversionError(report.message or "unsupported", report)

    if source_format == "qiskit" and target_format == "qd_algorithm_ir":
        return import_qiskit_circuit(source_obj)
    if source_format == "openqasm2" and target_format == "qd_algorithm_ir":
        return import_openqasm(source_obj)
    if source_format == "qd_algorithm_ir" and target_format == "qiskit":
        return export_qiskit(source_obj)
    if source_format == "qir":
        raise ConversionError(
            "QIR import is not implemented; see Adapter design in docs",
            ConversionReport.unsupported("qir", target_format, "stub only"),
        )
    raise ConversionError("Unhandled port path", report)
