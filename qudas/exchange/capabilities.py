"""変換 Capability 一覧と事前チェック。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from qudas.model.schema import ConversionKind
from qudas.model.conversion import ConversionReport


@dataclass
class ConversionCapability:
    source: str
    target: str
    kind: ConversionKind
    available: bool
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


_CAPABILITIES: List[ConversionCapability] = [
    ConversionCapability("qiskit", "qd_algorithm_ir", ConversionKind.LOSSLESS, True, "via QdAlgorithmIR.to_qiskit/from_qasm"),
    ConversionCapability("openqasm2", "qd_algorithm_ir", ConversionKind.LOSSLESS, True, "Qiskit QuantumCircuit.from_qasm_str"),
    ConversionCapability("openqasm3", "qd_algorithm_ir", ConversionKind.APPROXIMATE, False, "Use external compiler; not built-in"),
    ConversionCapability("qir", "qd_algorithm_ir", ConversionKind.UNSUPPORTED, False, "Adapter stub only; no parser"),
    ConversionCapability("qd_algorithm_ir", "qiskit", ConversionKind.LOSSLESS, True, "Supported gates subset"),
    ConversionCapability("qubo", "ising", ConversionKind.LOSSLESS, True, "via dimod BQM vartype change"),
    ConversionCapability("ising", "qubo", ConversionKind.LOSSLESS, True, "via dimod"),
    ConversionCapability("qubo", "dimod_bqm", ConversionKind.LOSSLESS, True, "QdAnnealingIR"),
    ConversionCapability("qubo", "amplify", ConversionKind.LOSSLESS, True, "requires amplify"),
    ConversionCapability("networkx", "qubo", ConversionKind.PROBLEM_LEVEL, True, "edge->quadratic; MaxCut uses dedicated mapping"),
    ConversionCapability("gate", "annealing", ConversionKind.UNSUPPORTED, False, "Arbitrary circuits cannot map to QUBO"),
    ConversionCapability("annealing", "gate", ConversionKind.UNSUPPORTED, False, "Use problem-level translation (e.g. QAOA)"),
    ConversionCapability("problem", "qubo", ConversionKind.PROBLEM_LEVEL, True, "via OptimizationProblem.to_model"),
]


def list_capabilities() -> List[Dict]:
    return [c.to_dict() for c in _CAPABILITIES]


def check_conversion(source: str, target: str) -> ConversionReport:
    source_l, target_l = source.lower(), target.lower()
    for c in _CAPABILITIES:
        if c.source == source_l and c.target == target_l:
            if not c.available or c.kind == ConversionKind.UNSUPPORTED:
                return ConversionReport.unsupported(
                    source_l,
                    target_l,
                    c.notes or "unsupported",
                    constraints=[c.kind.value],
                )
            return ConversionReport(
                kind=c.kind,
                source=source_l,
                target=target_l,
                success=True,
                message=c.notes,
            )
    return ConversionReport.unsupported(
        source_l, target_l, "No registered conversion capability"
    )
