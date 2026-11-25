"""変換の意味保存検証。

測定・条件分岐を含むプログラムにユニタリ比較だけを適用しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math


@dataclass
class SemanticCheckResult:
    name: str
    passed: bool
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _allclose(a, b, atol: float = 1e-6, rtol: float = 1e-5) -> bool:
    try:
        import numpy as np

        return bool(np.allclose(a, b, atol=atol, rtol=rtol))
    except Exception:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))
        return a == b


def verify_qubo_equivalence(
    qubo_a: Dict[Tuple[Any, Any], float],
    qubo_b: Dict[Tuple[Any, Any], float],
    *,
    constant_a: float = 0.0,
    constant_b: float = 0.0,
    atol: float = 1e-8,
) -> SemanticCheckResult:
    """QUBO 係数と定数項の一致。キーは正規化して比較。"""

    def norm(q: Dict) -> Dict[Tuple[str, str], float]:
        out: Dict[Tuple[str, str], float] = {}
        for (i, j), v in q.items():
            a, b = str(i), str(j)
            key = (a, a) if a == b else ((a, b) if a <= b else (b, a))
            out[key] = out.get(key, 0.0) + float(v)
        # ゼロ除去
        return {k: v for k, v in out.items() if abs(v) > atol}

    na, nb = norm(qubo_a), norm(qubo_b)
    keys = set(na) | set(nb)
    mismatches = []
    for k in sorted(keys):
        va, vb = na.get(k, 0.0), nb.get(k, 0.0)
        if abs(va - vb) > atol:
            mismatches.append({"key": list(k), "a": va, "b": vb})
    const_ok = abs(float(constant_a) - float(constant_b)) <= atol
    passed = not mismatches and const_ok
    return SemanticCheckResult(
        name="qubo_equivalence",
        passed=passed,
        message="ok" if passed else f"{len(mismatches)} coeff mismatches",
        details={
            "mismatches": mismatches,
            "constant_a": constant_a,
            "constant_b": constant_b,
            "constant_ok": const_ok,
        },
    )


def verify_ising_energy(
    linear: Dict[Any, float],
    quadratic: Dict[Tuple[Any, Any], float],
    offset: float,
    assignment: Dict[Any, int],
    expected_energy: float,
    *,
    atol: float = 1e-8,
) -> SemanticCheckResult:
    """Ising エネルギー E = offset + Σ h_i s_i + Σ J_ij s_i s_j。s∈{±1}。"""
    energy = float(offset)
    for i, h in linear.items():
        s = assignment.get(i, assignment.get(str(i), 0))
        # BINARY 誤混入を拒否しないが SPIN を期待
        if s in (0, 1):
            s = 1 if int(s) == 1 else -1
        energy += float(h) * float(s)
    for (i, j), J in quadratic.items():
        si = assignment.get(i, assignment.get(str(i), 0))
        sj = assignment.get(j, assignment.get(str(j), 0))
        if si in (0, 1):
            si = 1 if int(si) == 1 else -1
        if sj in (0, 1):
            sj = 1 if int(sj) == 1 else -1
        energy += float(J) * float(si) * float(sj)
    passed = abs(energy - float(expected_energy)) <= atol
    return SemanticCheckResult(
        name="ising_energy",
        passed=passed,
        message="ok" if passed else f"energy {energy} != {expected_energy}",
        details={"energy": energy, "expected": expected_energy},
    )


def verify_qubo_energy(
    qubo: Dict[Tuple[Any, Any], float],
    assignment: Dict[Any, int],
    expected_energy: float,
    *,
    constant: float = 0.0,
    atol: float = 1e-8,
) -> SemanticCheckResult:
    energy = float(constant)
    for (i, j), coeff in qubo.items():
        xi = int(assignment.get(i, assignment.get(str(i), 0)))
        xj = int(assignment.get(j, assignment.get(str(j), 0)))
        energy += float(coeff) * xi * xj
    passed = abs(energy - float(expected_energy)) <= atol
    return SemanticCheckResult(
        name="qubo_energy",
        passed=passed,
        message="ok" if passed else f"energy {energy} != {expected_energy}",
        details={"energy": energy, "expected": expected_energy},
    )


def _circuit_has_measure(circuit_or_ir: Any) -> bool:
    if hasattr(circuit_or_ir, "gates"):
        return any(getattr(g, "gate", "") == "measure" for g in circuit_or_ir.gates)
    data = getattr(circuit_or_ir, "data", None)
    if data is None:
        return False
    for item in data:
        inst = item[0] if isinstance(item, tuple) else getattr(item, "operation", item)
        name = getattr(inst, "name", "")
        if name == "measure":
            return True
    return False


def verify_unitary_equivalence(
    circuit_a: Any,
    circuit_b: Any,
    *,
    atol: float = 1e-6,
    ignore_global_phase: bool = True,
) -> SemanticCheckResult:
    """ユニタリ同値性。測定を含む場合は適用せず失敗メッセージを返す。"""
    if _circuit_has_measure(circuit_a) or _circuit_has_measure(circuit_b):
        return SemanticCheckResult(
            name="unitary_equivalence",
            passed=False,
            message="Circuits contain measurements; use verify_measurement_probabilities",
            details={"skipped": True},
        )
    try:
        import numpy as np
        from qiskit.quantum_info import Operator
    except ImportError as exc:
        return SemanticCheckResult(
            name="unitary_equivalence",
            passed=False,
            message=f"qiskit/numpy required: {exc}",
        )

    # IR → qiskit
    def to_qc(obj):
        if hasattr(obj, "to_qiskit"):
            return obj.to_qiskit()
        return obj

    ua = Operator(to_qc(circuit_a)).data
    ub = Operator(to_qc(circuit_b)).data
    if ua.shape != ub.shape:
        return SemanticCheckResult(
            name="unitary_equivalence",
            passed=False,
            message=f"shape mismatch {ua.shape} vs {ub.shape}",
        )

    if ignore_global_phase:
        # U† V ≈ e^{iφ} I
        prod = ua.conj().T @ ub
        # 対角成分の位相を推定
        diag = np.diag(prod)
        # ゼロ回避
        nz = diag[np.abs(diag) > atol]
        if len(nz) == 0:
            phase = 1.0 + 0j
        else:
            phase = nz[0] / abs(nz[0])
        aligned = prod / phase
        passed = bool(np.allclose(aligned, np.eye(aligned.shape[0]), atol=atol))
        return SemanticCheckResult(
            name="unitary_equivalence",
            passed=passed,
            message="ok (global phase ignored)" if passed else "unitaries differ",
            details={"ignore_global_phase": True},
        )

    passed = bool(np.allclose(ua, ub, atol=atol))
    return SemanticCheckResult(
        name="unitary_equivalence",
        passed=passed,
        message="ok" if passed else "unitaries differ",
        details={"ignore_global_phase": False},
    )


def verify_measurement_probabilities(
    circuit_a: Any,
    circuit_b: Any,
    *,
    shots: int = 4096,
    seed: int = 42,
    atol: float = 0.05,
) -> SemanticCheckResult:
    """測定確率分布の近似一致（シミュレータ）。"""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except ImportError as exc:
        return SemanticCheckResult(
            name="measurement_probabilities",
            passed=False,
            message=f"qiskit required: {exc}",
        )

    def to_qc(obj):
        if hasattr(obj, "to_qiskit"):
            return obj.to_qiskit()
        return obj

    def probs(qc) -> Dict[str, float]:
        # 測定を除いたユニタリ部分の状態ベクトルから Z 基底確率
        qc2 = qc.copy()
        # measure 除去
        new = QuantumCircuit(qc2.num_qubits)
        for item in qc2.data:
            inst = item[0] if isinstance(item, tuple) else item.operation
            qargs = item[1] if isinstance(item, tuple) else item.qubits
            if getattr(inst, "name", "") == "measure":
                continue
            new.append(inst, qargs)
        sv = Statevector.from_instruction(new)
        p = sv.probabilities_dict()
        return {k: float(v) for k, v in p.items()}

    pa, pb = probs(to_qc(circuit_a)), probs(to_qc(circuit_b))
    keys = set(pa) | set(pb)
    max_diff = 0.0
    for k in keys:
        max_diff = max(max_diff, abs(pa.get(k, 0.0) - pb.get(k, 0.0)))
    passed = max_diff <= atol
    return SemanticCheckResult(
        name="measurement_probabilities",
        passed=passed,
        message="ok" if passed else f"max_diff={max_diff}",
        details={"max_diff": max_diff, "atol": atol, "shots_unused": shots, "seed": seed},
    )


def verify_pipeline_types(
    model: Any,
    registry: Any = None,
) -> SemanticCheckResult:
    """Pipeline Model の type_id 解決と入出力スキーマの存在確認。"""
    errors: List[str] = []
    if registry is None:
        from qudas.pipeline.registry import default_registry

        registry = default_registry
    for n in getattr(model, "nodes", []):
        if not registry.has(n.type_id):
            errors.append(f"unregistered type_id={n.type_id}")
        # 演算メタ
        meta = getattr(n, "metadata", {}) or {}
        if "op_version" in meta and not isinstance(meta["op_version"], str):
            errors.append(f"node {n.id}: op_version must be str")
    if getattr(model, "input_schema", None) is not None and not isinstance(
        model.input_schema, dict
    ):
        errors.append("input_schema must be dict")
    if getattr(model, "output_schema", None) is not None and not isinstance(
        model.output_schema, dict
    ):
        errors.append("output_schema must be dict")
    return SemanticCheckResult(
        name="pipeline_types",
        passed=not errors,
        message="ok" if not errors else "; ".join(errors),
        details={"errors": errors},
    )
