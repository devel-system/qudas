"""JSON 交換のルート Envelope。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .schema import SCHEMA_VERSION, ModelType


@dataclass
class QdModelEnvelope:
    """Qudas モデル交換の共通ルート。

    任意の Python コードは含まない。payload は JSON 互換 dict のみ。
    """

    model_type: ModelType
    payload: Dict[str, Any]
    schema_version: str = SCHEMA_VERSION
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    backend_requirements: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    domain_model_refs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        mt = self.model_type.value if isinstance(self.model_type, ModelType) else self.model_type
        return {
            "schema_version": self.schema_version,
            "model_type": mt,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "domain_model_refs": self.domain_model_refs,
            "parameters": self.parameters,
            "backend_requirements": self.backend_requirements,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "provenance": self.provenance,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QdModelEnvelope":
        if not isinstance(data, dict):
            raise TypeError("Envelope data must be a dict")
        if "model_type" not in data or "payload" not in data:
            raise ValueError("Envelope requires 'model_type' and 'payload'")
        schema_version = data.get("schema_version", SCHEMA_VERSION)
        if schema_version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(
                f"Unsupported schema major version: {schema_version} "
                f"(current={SCHEMA_VERSION})"
            )
        mt = data["model_type"]
        if isinstance(mt, str):
            mt = ModelType(mt)
        return cls(
            schema_version=schema_version,
            model_type=mt,
            payload=dict(data["payload"]),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            parameters=dict(data.get("parameters") or {}),
            backend_requirements=list(data.get("backend_requirements") or []),
            dependencies=list(data.get("dependencies") or []),
            metadata=dict(data.get("metadata") or {}),
            provenance=dict(data.get("provenance") or {}),
            domain_model_refs=list(data.get("domain_model_refs") or []),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "QdModelEnvelope":
        return cls.from_dict(json.loads(text))

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "QdModelEnvelope":
        path = Path(path)
        return cls.from_json(path.read_text(encoding="utf-8"))
