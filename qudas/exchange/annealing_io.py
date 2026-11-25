"""Annealing Domain の Port（既存 QdAnnealingIR を利用）。"""

from __future__ import annotations

from typing import Any

from qudas.model.conversion import ConversionError, ConversionReport
from qudas.model.schema import ConversionKind
from .capabilities import check_conversion


def port_annealing(source_obj: Any, source_format: str, target_format: str):
    """QUBO / Ising / dimod / amplify 等の変換。"""
    src, tgt = source_format.lower(), target_format.lower()

    # Gate↔Annealing の直変換を拒否
    if src in ("gate", "qiskit", "openqasm2", "qd_algorithm_ir") and tgt in (
        "qubo",
        "ising",
        "annealing",
    ):
        report = ConversionReport.unsupported(
            src,
            tgt,
            "Arbitrary gate circuits cannot be converted to annealing models. "
            "Use problem-level translation (e.g. MaxCut → QUBO → QAOA).",
            constraints=["use problem_level"],
        )
        raise ConversionError(report.message, report)

    report = check_conversion(src, tgt)
    if report.kind == ConversionKind.UNSUPPORTED or not report.success:
        # networkx→qubo は capability 上 PROBLEM_LEVEL だが AnnealingIR.from_networkx はある
        if not (src == "networkx" and tgt == "qubo"):
            if not report.success:
                raise ConversionError(report.message or "unsupported", report)

    from qudas.annealing import QdAnnealingIR

    ir = QdAnnealingIR()
    if src == "qubo" and isinstance(source_obj, dict):
        ir = QdAnnealingIR(qubo=source_obj)
    elif src == "dimod_bqm":
        ir = QdAnnealingIR().from_dimod_bqm(source_obj)
    elif src == "networkx":
        ir = QdAnnealingIR().from_networkx(source_obj)
        report = ConversionReport(
            kind=ConversionKind.PROBLEM_LEVEL,
            source=src,
            target=tgt,
            success=True,
            message="Edges mapped to quadratic terms (not MaxCut objective)",
        )
    elif src == "amplify":
        ir = QdAnnealingIR().from_amplify(source_obj)
    elif hasattr(source_obj, "qubo"):
        ir = QdAnnealingIR(qubo=dict(source_obj.qubo))
    else:
        raise ConversionError(f"Cannot interpret source as {src}", report)

    if tgt == "qubo":
        return dict(ir.qubo), report
    if tgt == "annealing_ir":
        return ir, report
    if tgt == "dimod_bqm":
        return ir.to_dimod_bqm(), report
    if tgt == "ising":
        bqm = ir.to_dimod_bqm()
        ising = bqm.change_vartype("SPIN", inplace=False)
        return {
            "linear": dict(ising.linear),
            "quadratic": {tuple(k): float(v) for k, v in ising.quadratic.items()},
            "offset": float(ising.offset),
        }, report
    if tgt == "amplify":
        return ir.to_amplify(), report

    raise ConversionError(f"Unhandled annealing target {tgt}", report)
