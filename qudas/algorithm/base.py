"""Algorithm 仕様（実装 Pipeline とは分離）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from qudas.pipeline.model import QdPipelineModel


@dataclass
class AlgorithmSpec:
    """アルゴリズムの宣言的仕様。"""

    algorithm_id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_schema: Optional[Dict[str, Any]] = None
    required_dependencies: List[str] = field(default_factory=list)
    compatible_problems: List[str] = field(default_factory=list)
    backend_requirements: List[str] = field(default_factory=list)
    paper: Dict[str, Any] = field(default_factory=dict)
    # 実装構造（Pipeline Model）。未設定なら build_pipeline で生成
    pipeline_model: Optional[QdPipelineModel] = None

    def build_pipeline(self, **overrides: Any) -> QdPipelineModel:
        """パラメータを反映した Pipeline Model を返す。"""
        if self.pipeline_model is None:
            raise NotImplementedError(
                f"{self.algorithm_id} has no pipeline_model; override build_pipeline"
            )
        model = QdPipelineModel.from_dict(self.pipeline_model.to_dict())
        params = dict(self.parameters)
        params.update(overrides)
        model.parameters.update(params)
        return model

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "parameters": self.parameters,
            "parameter_schema": self.parameter_schema,
            "required_dependencies": self.required_dependencies,
            "compatible_problems": self.compatible_problems,
            "backend_requirements": self.backend_requirements,
            "paper": self.paper,
            "pipeline_model": self.pipeline_model.to_dict() if self.pipeline_model else None,
        }
