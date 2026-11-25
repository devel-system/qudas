"""論文・実験条件を含む Algorithm Package。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from qudas.model.schema import SCHEMA_VERSION, ModelType
from qudas.model.envelope import QdModelEnvelope
from qudas.pipeline.model import QdPipelineModel
from qudas.result.common import QdExperimentResult, Timer, TimingBreakdown

from .base import AlgorithmSpec


@dataclass
class AlgorithmPackage:
    """再現・再利用のためのパッケージ。

    第三者コードのライセンスが不明な場合はコードを同梱せず、
    source_repository / commit / license の参照のみを保持する。
    """

    algorithm_id: str
    name: str
    version: str = "0.1.0"
    paper: Dict[str, Any] = field(default_factory=dict)
    source_repository: Optional[str] = None
    source_commit: Optional[str] = None
    license: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_dependencies: List[str] = field(default_factory=list)
    compatible_problems: List[str] = field(default_factory=list)
    pipeline_model: Optional[QdPipelineModel] = None
    tests: Dict[str, Any] = field(default_factory=dict)
    reference_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 実行ディスパッチ用（シリアライズされないランタイムフック名）
    runner_id: Optional[str] = None

    @classmethod
    def from_spec(cls, spec: AlgorithmSpec, **kwargs: Any) -> "AlgorithmPackage":
        pipe = kwargs.pop("pipeline_model", None) or (
            spec.build_pipeline() if spec.pipeline_model else None
        )
        paper = kwargs.pop("paper", None)
        return cls(
            algorithm_id=spec.algorithm_id,
            name=kwargs.pop("name", spec.name),
            version=kwargs.pop("version", spec.version),
            paper=dict(paper if paper is not None else spec.paper),
            input_schema=kwargs.pop("input_schema", spec.input_schema),
            output_schema=kwargs.pop("output_schema", spec.output_schema),
            parameters=dict(kwargs.pop("parameters", spec.parameters)),
            required_dependencies=list(
                kwargs.pop("required_dependencies", spec.required_dependencies)
            ),
            compatible_problems=list(
                kwargs.pop("compatible_problems", spec.compatible_problems)
            ),
            pipeline_model=pipe,
            runner_id=kwargs.pop("runner_id", spec.algorithm_id),
            metadata=kwargs.pop("metadata", {}),
            source_repository=kwargs.pop("source_repository", None),
            source_commit=kwargs.pop("source_commit", None),
            license=kwargs.pop("license", None),
            tests=kwargs.pop("tests", {}),
            reference_results=kwargs.pop("reference_results", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "algorithm_id": self.algorithm_id,
            "name": self.name,
            "version": self.version,
            "paper": self.paper,
            "source_repository": self.source_repository,
            "source_commit": self.source_commit,
            "license": self.license,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "parameters": self.parameters,
            "required_dependencies": self.required_dependencies,
            "compatible_problems": self.compatible_problems,
            "pipeline_model": self.pipeline_model.to_dict() if self.pipeline_model else None,
            "tests": self.tests,
            "reference_results": self.reference_results,
            "metadata": self.metadata,
            "runner_id": self.runner_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlgorithmPackage":
        pipe = data.get("pipeline_model")
        return cls(
            algorithm_id=data["algorithm_id"],
            name=data.get("name", data["algorithm_id"]),
            version=str(data.get("version", "0.1.0")),
            paper=dict(data.get("paper") or {}),
            source_repository=data.get("source_repository"),
            source_commit=data.get("source_commit"),
            license=data.get("license"),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            parameters=dict(data.get("parameters") or {}),
            required_dependencies=list(data.get("required_dependencies") or []),
            compatible_problems=list(data.get("compatible_problems") or []),
            pipeline_model=QdPipelineModel.from_dict(pipe) if pipe else None,
            tests=dict(data.get("tests") or {}),
            reference_results=list(data.get("reference_results") or []),
            metadata=dict(data.get("metadata") or {}),
            runner_id=data.get("runner_id"),
        )

    def to_envelope(self) -> QdModelEnvelope:
        return QdModelEnvelope(
            model_type=ModelType.ALGORITHM_PACKAGE,
            payload=self.to_dict(),
            parameters=self.parameters,
            dependencies=self.required_dependencies,
            metadata={"name": self.name, "version": self.version},
            provenance={
                "paper": self.paper,
                "source_repository": self.source_repository,
                "source_commit": self.source_commit,
                "license": self.license,
            },
        )

    def save(self, path: Union[str, Path]) -> None:
        self.to_envelope().save(path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AlgorithmPackage":
        env = QdModelEnvelope.load(path)
        if env.model_type != ModelType.ALGORITHM_PACKAGE:
            raise ValueError(f"Expected algorithm_package, got {env.model_type}")
        return cls.from_dict(env.payload)

    def run(self, problem: Any, **overrides: Any) -> QdExperimentResult:
        """問題データを差し替えて同一 Algorithm を実行する。"""
        from .runners import run_algorithm

        params = dict(self.parameters)
        params.update(overrides)
        runner = self.runner_id or self.algorithm_id
        return run_algorithm(runner, problem, params, package=self)
