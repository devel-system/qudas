"""方式横断で比較可能な実験 Result。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from qudas.model.schema import SCHEMA_VERSION, ModelType
from qudas.model.envelope import QdModelEnvelope


@dataclass
class TimingBreakdown:
    """時間指標を混同しないための分解。

    ゲートの compile / QPU / 古典最適化と、アニーリングの待機時間は別フィールド。
    """

    total_s: Optional[float] = None
    compile_s: Optional[float] = None
    qpu_s: Optional[float] = None
    classical_optimize_s: Optional[float] = None
    queue_wait_s: Optional[float] = None
    anneal_s: Optional[float] = None
    other_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TimingBreakdown":
        data = data or {}
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


@dataclass
class ResourceInfo:
    """計算方式ごとに意味が異なる資源情報（ゲート数とアニーリング資源を混同しない）。"""

    gate_count: Optional[int] = None
    circuit_depth: Optional[int] = None
    num_qubits: Optional[int] = None
    num_variables: Optional[int] = None
    shots: Optional[int] = None
    num_reads: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d.get("extra"):
            d.pop("extra", None)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ResourceInfo":
        data = data or {}
        known = {k: data.get(k) for k in ("gate_count", "circuit_depth", "num_qubits", "num_variables", "shots", "num_reads")}
        extra = dict(data.get("extra") or {})
        for k, v in data.items():
            if k not in known and k != "extra":
                extra[k] = v
        return cls(**known, extra=extra)


@dataclass
class QdExperimentResult:
    """共通 Result 仕様。"""

    solution: Any = None
    objective_value: Optional[float] = None
    feasibility: Optional[bool] = None
    sampling_distribution: Optional[Dict[str, Any]] = None
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)
    backend: Optional[str] = None
    algorithm: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    random_seed: Optional[int] = None
    resources: ResourceInfo = field(default_factory=ResourceInfo)
    status: str = "ok"
    message: str = ""
    # 最小化目的（QUBO energy 等）と問題固有の再評価値を分離
    minimized_energy: Optional[float] = None
    problem_objective: Optional[float] = None
    sense: str = "maximize"  # maximize | minimize
    provenance: Dict[str, Any] = field(default_factory=dict)
    version: str = SCHEMA_VERSION
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "solution": self.solution,
            "objective_value": self.objective_value,
            "feasibility": self.feasibility,
            "sampling_distribution": self.sampling_distribution,
            "timing": self.timing.to_dict(),
            "backend": self.backend,
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "random_seed": self.random_seed,
            "resources": self.resources.to_dict(),
            "status": self.status,
            "message": self.message,
            "minimized_energy": self.minimized_energy,
            "problem_objective": self.problem_objective,
            "sense": self.sense,
            "provenance": self.provenance,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QdExperimentResult":
        return cls(
            version=str(data.get("version", SCHEMA_VERSION)),
            solution=data.get("solution"),
            objective_value=data.get("objective_value"),
            feasibility=data.get("feasibility"),
            sampling_distribution=data.get("sampling_distribution"),
            timing=TimingBreakdown.from_dict(data.get("timing")),
            backend=data.get("backend"),
            algorithm=data.get("algorithm"),
            parameters=dict(data.get("parameters") or {}),
            random_seed=data.get("random_seed"),
            resources=ResourceInfo.from_dict(data.get("resources")),
            status=str(data.get("status", "ok")),
            message=str(data.get("message", "")),
            minimized_energy=data.get("minimized_energy"),
            problem_objective=data.get("problem_objective"),
            sense=str(data.get("sense", "maximize")),
            provenance=dict(data.get("provenance") or {}),
            raw=dict(data.get("raw") or {}),
        )

    def to_envelope(self) -> QdModelEnvelope:
        return QdModelEnvelope(
            model_type=ModelType.RESULT,
            payload=self.to_dict(),
            provenance=self.provenance,
            metadata={"algorithm": self.algorithm, "backend": self.backend},
        )

    def save(self, path: Union[str, Path]) -> None:
        self.to_envelope().save(path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "QdExperimentResult":
        env = QdModelEnvelope.load(path)
        if env.model_type != ModelType.RESULT:
            raise ValueError(f"Expected result envelope, got {env.model_type}")
        return cls.from_dict(env.payload)

    @classmethod
    def timed(cls, **kwargs: Any) -> "QdExperimentResult":
        """ヘルパ: total_s を後から埋める用の開始時刻付き生成は呼び出し側で行う。"""
        return cls(**kwargs)


class Timer:
    """簡易壁時計。"""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0
