"""変換 Capability 一覧と事前チェック（ConversionGraph と同期）。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from qudas.model.schema import ConversionKind
from qudas.model.conversion import ConversionReport
from qudas.model.graph import ConversionGraph, default_conversion_graph, get_default_graph


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


# 宣言的 Capability（Graph 初期化のソース・オブ・トゥルース）
_CAPABILITY_SPECS: List[ConversionCapability] = [
    ConversionCapability("qiskit", "qd_algorithm_ir", ConversionKind.LOSSLESS, True, "via QdAlgorithmIR.from_qasm"),
    ConversionCapability("openqasm2", "qd_algorithm_ir", ConversionKind.LOSSLESS, True, "Qiskit QuantumCircuit.from_qasm_str"),
    ConversionCapability("openqasm2", "qiskit", ConversionKind.LOSSLESS, True, "Qiskit from_qasm_str"),
    ConversionCapability("qd_algorithm_ir", "qiskit", ConversionKind.LOSSLESS, True, "Supported gates subset"),
    ConversionCapability("qd_algorithm_ir", "openqasm2", ConversionKind.LOSSLESS, True, "via to_qiskit().qasm()"),
    ConversionCapability("qiskit", "openqasm2", ConversionKind.LOSSLESS, True, "QuantumCircuit.qasm()"),
    ConversionCapability("openqasm3", "qiskit", ConversionKind.APPROXIMATE, True, "Qiskit qasm3 loads; subset only"),
    ConversionCapability("openqasm3", "qd_algorithm_ir", ConversionKind.APPROXIMATE, True, "via openqasm3->qiskit->IR"),
    ConversionCapability("qir", "qd_algorithm_ir", ConversionKind.UNSUPPORTED, False, "Adapter contract only; no parser"),
    ConversionCapability("qubo", "ising", ConversionKind.LOSSLESS, True, "via dimod BQM vartype change"),
    ConversionCapability("ising", "qubo", ConversionKind.LOSSLESS, True, "via dimod"),
    ConversionCapability("qubo", "dimod_bqm", ConversionKind.LOSSLESS, True, "QdAnnealingIR"),
    ConversionCapability("dimod_bqm", "qubo", ConversionKind.LOSSLESS, True, "BQM to dict"),
    ConversionCapability("qubo", "annealing_ir", ConversionKind.LOSSLESS, True, "QdAnnealingIR(qubo=...)"),
    ConversionCapability("annealing_ir", "qubo", ConversionKind.LOSSLESS, True, "QdAnnealingIR.qubo"),
    ConversionCapability("annealing_ir", "dimod_bqm", ConversionKind.LOSSLESS, True, "to_dimod_bqm"),
    ConversionCapability("qubo", "amplify", ConversionKind.LOSSLESS, True, "requires amplify"),
    ConversionCapability("networkx", "qubo", ConversionKind.PROBLEM_LEVEL, True, "edge->quadratic; MaxCut dedicated"),
    ConversionCapability("pyqubo", "qubo", ConversionKind.LOSSLESS, True, "CompiledModel / QuData"),
    ConversionCapability("gate", "annealing", ConversionKind.UNSUPPORTED, False, "Arbitrary circuits cannot map to QUBO"),
    ConversionCapability("annealing", "gate", ConversionKind.UNSUPPORTED, False, "Use problem-level translation (e.g. QAOA)"),
    ConversionCapability("problem", "qubo", ConversionKind.PROBLEM_LEVEL, True, "via OptimizationProblem.to_model"),
]


# OpenQASM3 機能単位 Capability（パーサが扱える範囲の宣言）
OPENQASM3_FEATURE_CAPABILITIES: Dict[str, Dict[str, object]] = {
    "basic_gates": {"supported": True, "notes": "h,x,cx,rz,rx,ry via IR subset"},
    "parameterized_gates": {"supported": True, "notes": "numeric params only"},
    "measurements": {"supported": True, "notes": "measure -> IR measure"},
    "reset": {"supported": False, "notes": "not mapped in QdGateIR"},
    "classical_control": {"supported": False, "notes": "if/while not imported"},
    "dynamic_circuits": {"supported": False, "notes": "mid-circuit feedback unsupported"},
}


def _identity(x):
    return x


def _build_converters():
    """遅延 import を含む converter 群。"""

    def openqasm2_to_qiskit(qasm: str):
        from qiskit import QuantumCircuit

        return QuantumCircuit.from_qasm_str(qasm)

    def qiskit_to_openqasm2(qc):
        if hasattr(qc, "qasm"):
            return qc.qasm()
        # Qiskit 1.x
        from qiskit.qasm2 import dumps

        return dumps(qc)

    def qiskit_to_ir(qc):
        from qudas.gate import QdAlgorithmIR

        return QdAlgorithmIR.from_qasm(qc)

    def ir_to_qiskit(ir):
        return ir.to_qiskit()

    def ir_to_openqasm2(ir):
        return qiskit_to_openqasm2(ir.to_qiskit())

    def openqasm2_to_ir(qasm: str):
        return qiskit_to_ir(openqasm2_to_qiskit(qasm))

    def openqasm3_to_qiskit(qasm: str):
        from qudas.exchange.gate_io import loads_openqasm3

        return loads_openqasm3(qasm)

    def openqasm3_to_ir(qasm: str):
        return qiskit_to_ir(openqasm3_to_qiskit(qasm))

    def qubo_to_annealing_ir(qubo):
        from qudas.annealing import QdAnnealingIR

        return QdAnnealingIR(qubo=dict(qubo))

    def annealing_ir_to_qubo(ir):
        return dict(ir.qubo)

    def qubo_to_dimod(qubo):
        return qubo_to_annealing_ir(qubo).to_dimod_bqm()

    def dimod_to_qubo(bqm):
        # linear + quadratic → dict[(i,j)]
        out = {}
        for v, bias in bqm.linear.items():
            out[(v, v)] = float(bias)
        for (u, v), bias in bqm.quadratic.items():
            a, b = (u, v) if str(u) <= str(v) else (v, u)
            out[(a, b)] = float(bias)
        return out

    def annealing_ir_to_dimod(ir):
        return ir.to_dimod_bqm()

    def qubo_to_ising(qubo):
        bqm = qubo_to_dimod(qubo)
        ising = bqm.change_vartype("SPIN", inplace=False)
        return {
            "linear": dict(ising.linear),
            "quadratic": {tuple(k): float(v) for k, v in ising.quadratic.items()},
            "offset": float(ising.offset),
            "vartype": "SPIN",
        }

    def ising_to_qubo(ising: dict):
        import dimod

        bqm = dimod.BinaryQuadraticModel(
            ising.get("linear") or {},
            {tuple(k): v for k, v in (ising.get("quadratic") or {}).items()},
            float(ising.get("offset") or 0.0),
            vartype=dimod.SPIN,
        )
        qubo_bqm = bqm.change_vartype("BINARY", inplace=False)
        return dimod_to_qubo(qubo_bqm)

    def problem_to_qubo(problem):
        return problem.to_model("qubo")

    def networkx_to_qubo(graph):
        from qudas.problem.maxcut import MaxCutProblem

        return MaxCutProblem.from_networkx(graph).to_model("qubo")

    def qubo_to_amplify(qubo):
        from qudas.annealing import QdAnnealingIR

        return QdAnnealingIR(qubo=dict(qubo)).to_amplify()

    def pyqubo_to_qubo(model):
        # CompiledModel or Express
        if hasattr(model, "compile"):
            compiled = model.compile()
        else:
            compiled = model
        bqm = compiled.to_bqm()
        return dimod_to_qubo(bqm)

    return {
        ("openqasm2", "qiskit"): openqasm2_to_qiskit,
        ("qiskit", "openqasm2"): qiskit_to_openqasm2,
        ("qiskit", "qd_algorithm_ir"): qiskit_to_ir,
        ("qd_algorithm_ir", "qiskit"): ir_to_qiskit,
        ("qd_algorithm_ir", "openqasm2"): ir_to_openqasm2,
        ("openqasm2", "qd_algorithm_ir"): openqasm2_to_ir,
        ("openqasm3", "qiskit"): openqasm3_to_qiskit,
        ("openqasm3", "qd_algorithm_ir"): openqasm3_to_ir,
        ("qubo", "annealing_ir"): qubo_to_annealing_ir,
        ("annealing_ir", "qubo"): annealing_ir_to_qubo,
        ("qubo", "dimod_bqm"): qubo_to_dimod,
        ("dimod_bqm", "qubo"): dimod_to_qubo,
        ("annealing_ir", "dimod_bqm"): annealing_ir_to_dimod,
        ("qubo", "ising"): qubo_to_ising,
        ("ising", "qubo"): ising_to_qubo,
        ("problem", "qubo"): problem_to_qubo,
        ("networkx", "qubo"): networkx_to_qubo,
        ("qubo", "amplify"): qubo_to_amplify,
        ("pyqubo", "qubo"): pyqubo_to_qubo,
        ("gate", "annealing"): None,
        ("annealing", "gate"): None,
        ("qir", "qd_algorithm_ir"): None,
    }


_GRAPH_READY = False


def ensure_graph_initialized(graph: Optional[ConversionGraph] = None) -> ConversionGraph:
    """Capability 仕様を Graph へ登録（一度だけ）。"""
    global _GRAPH_READY
    graph = graph or get_default_graph()
    if graph is default_conversion_graph and _GRAPH_READY and graph.list_edges():
        return graph

    converters = _build_converters()
    for cap in _CAPABILITY_SPECS:
        key = (cap.source, cap.target)
        conv = converters.get(key)
        if graph.has_edge(cap.source, cap.target):
            # overwrite sync
            graph.register(
                cap.source,
                cap.target,
                cap.kind,
                converter=conv,
                notes=cap.notes,
                available=cap.available,
                overwrite=True,
            )
        else:
            graph.register(
                cap.source,
                cap.target,
                cap.kind,
                converter=conv,
                notes=cap.notes,
                available=cap.available,
                overwrite=True,
            )
    if graph is default_conversion_graph:
        _GRAPH_READY = True
    return graph


def list_capabilities() -> List[Dict]:
    ensure_graph_initialized()
    return [c.to_dict() for c in _CAPABILITY_SPECS]


def list_openqasm3_features() -> Dict[str, Dict[str, object]]:
    return dict(OPENQASM3_FEATURE_CAPABILITIES)


def check_conversion(source: str, target: str) -> ConversionReport:
    graph = ensure_graph_initialized()
    return graph.can_convert(source, target)


def convert(obj, source: str, target: str, prefer: str = "shortest"):
    """ConversionGraph 経由の変換。"""
    graph = ensure_graph_initialized()
    return graph.convert(obj, source, target, prefer=prefer)
