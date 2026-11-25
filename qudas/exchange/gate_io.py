"""Gate Domain の Import / Port / OpenQASM。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from qudas.model.conversion import ConversionError, ConversionReport
from qudas.model.schema import ConversionKind
from .capabilities import (
    check_conversion,
    convert as graph_convert,
    ensure_graph_initialized,
    list_openqasm3_features,
)


def loads_openqasm3(qasm: str):
    """OpenQASM 3 文字列 → Qiskit QuantumCircuit（外部パーサ優先）。

    対応範囲は capabilities.OPENQASM3_FEATURE_CAPABILITIES を参照。
    Classical control / dynamic circuits は拒否または警告対象。
    """
    qasm_l = qasm.lstrip().lower()
    if "openqasm 3" not in qasm_l and "openqasm3" not in qasm_l:
        # 明示バージョン無しでも qasm3 ローダを試す
        pass

    # 未対応構文の事前チェック（厳密パーサではない）
    forbidden = []
    features = list_openqasm3_features()
    for token, feat in (
        ("reset ", "reset"),
        ("if (", "classical_control"),
        ("while ", "classical_control"),
        ("switch ", "classical_control"),
    ):
        if token in qasm_l and not features.get(feat, {}).get("supported"):
            forbidden.append(feat)
    if forbidden:
        raise ConversionError(
            f"OpenQASM3 features not supported: {forbidden}",
            ConversionReport(
                kind=ConversionKind.UNSUPPORTED,
                source="openqasm3",
                target="qiskit",
                success=False,
                message=f"unsupported features: {forbidden}",
                constraints=forbidden,
            ),
        )

    try:
        from qiskit import qasm3  # type: ignore

        return qasm3.loads(qasm)
    except ImportError as exc:
        raise ConversionError(
            "qiskit.qasm3 is required for OpenQASM3 import",
            ConversionReport.unsupported(
                "openqasm3", "qiskit", "install qiskit with qasm3 support"
            ),
        ) from exc
    except Exception as exc:
        raise ConversionError(
            f"Failed to parse OpenQASM3: {exc}",
            ConversionReport(
                kind=ConversionKind.APPROXIMATE,
                source="openqasm3",
                target="qiskit",
                success=False,
                message=str(exc),
            ),
        ) from exc


def import_qiskit_circuit(circuit: Any):
    """Qiskit QuantumCircuit → QdAlgorithmIR。"""
    ensure_graph_initialized()
    return graph_convert(circuit, "qiskit", "qd_algorithm_ir")


def import_openqasm(qasm: str, version: Optional[int] = None):
    """OpenQASM 文字列 → QdAlgorithmIR。

    version=2|3。未指定時はヘッダから推定（既定 2）。
    """
    ensure_graph_initialized()
    text = qasm if isinstance(qasm, str) else str(qasm)
    header = text.lstrip()[:64].lower()
    if version is None:
        if "openqasm 3" in header or header.startswith("openqasm3"):
            version = 3
        else:
            version = 2

    if version == 3:
        return graph_convert(text, "openqasm3", "qd_algorithm_ir")
    if version == 2:
        try:
            return graph_convert(text, "openqasm2", "qd_algorithm_ir")
        except ConversionError:
            # ヘッダ誤認の救済: 3 を試す
            if "openqasm 3" in header:
                return graph_convert(text, "openqasm3", "qd_algorithm_ir")
            raise
    raise ConversionError(
        f"Unsupported OpenQASM version: {version}",
        ConversionReport.unsupported("openqasm", "qd_algorithm_ir", f"version={version}"),
    )


def export_qiskit(algorithm_ir: Any):
    """QdAlgorithmIR → Qiskit QuantumCircuit。"""
    ensure_graph_initialized()
    try:
        return graph_convert(algorithm_ir, "qd_algorithm_ir", "qiskit")
    except ConversionError:
        raise
    except ValueError as exc:
        approx = ConversionReport(
            kind=ConversionKind.UNSUPPORTED,
            source="qd_algorithm_ir",
            target="qiskit",
            success=False,
            message=str(exc),
            constraints=["supported_gates: x,h,cx,rz,rx,ry,measure"],
        )
        raise ConversionError(str(exc), approx) from exc


def export_openqasm2(algorithm_ir: Any) -> Tuple[str, ConversionReport]:
    ensure_graph_initialized()
    return graph_convert(algorithm_ir, "qd_algorithm_ir", "openqasm2")


def port_gate(source_obj: Any, source_format: str, target_format: str):
    """Gate 系の Port（ConversionGraph）。"""
    ensure_graph_initialized()
    source_format = source_format.lower()
    target_format = target_format.lower()
    if source_format == "qir":
        raise ConversionError(
            "QIR import is not implemented; see Adapter design in docs",
            ConversionReport.unsupported("qir", target_format, "stub only"),
        )
    return graph_convert(source_obj, source_format, target_format)


def openqasm3_capability_report() -> Dict[str, Any]:
    return {
        "format": "openqasm3",
        "features": list_openqasm3_features(),
        "conversion": check_conversion("openqasm3", "qd_algorithm_ir").to_dict(),
    }
