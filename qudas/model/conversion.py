"""変換の可否・制約を表すレポート。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .schema import ConversionKind


class ConversionError(ValueError):
    """変換不能または不正な変換要求。"""

    def __init__(self, message: str, report: Optional["ConversionReport"] = None):
        super().__init__(message)
        self.report = report


@dataclass
class ConversionReport:
    kind: ConversionKind
    source: str
    target: str
    success: bool
    message: str = ""
    constraints: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value if isinstance(self.kind, ConversionKind) else self.kind
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversionReport":
        kind = data.get("kind", ConversionKind.UNSUPPORTED)
        if isinstance(kind, str):
            kind = ConversionKind(kind)
        return cls(
            kind=kind,
            source=data.get("source", ""),
            target=data.get("target", ""),
            success=bool(data.get("success", False)),
            message=data.get("message", ""),
            constraints=list(data.get("constraints") or []),
            details=dict(data.get("details") or {}),
        )

    @classmethod
    def unsupported(
        cls, source: str, target: str, message: str, constraints: Optional[List[str]] = None
    ) -> "ConversionReport":
        return cls(
            kind=ConversionKind.UNSUPPORTED,
            source=source,
            target=target,
            success=False,
            message=message,
            constraints=constraints or [],
        )
