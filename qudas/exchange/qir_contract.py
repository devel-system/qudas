"""QIR Adapter Contract（実装なし・次期切り分け用）。

独自 LLVM コンパイラは開発しない。既存実行基盤（例: QIR runner /
PyQIR / ベンダー提供ツール）を利用できる場合のみ Adapter を追加する。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List


@dataclass
class QirAdapterContract:
    """QIR 連携の契約と技術的制約。"""

    format_id: str = "qir"
    status: str = "not_implemented"
    import_supported: bool = False
    export_supported: bool = False
    requires: List[str] = field(
        default_factory=lambda: [
            "External QIR bitstream or LLVM bitcode reader",
            "Mapping from QIR profiles (Base / Adaptive) to QdGateIR subset",
        ]
    )
    constraints: List[str] = field(
        default_factory=lambda: [
            "No in-tree LLVM toolchain",
            "No guarantee of round-trip with OpenQASM3 dynamic circuits",
            "Measurement / classical feedback mapping is out of scope for v0 PoC",
        ]
    )
    proposed_api: Dict[str, str] = field(
        default_factory=lambda: {
            "import_qir": "bytes|path -> (QdAlgorithmIR, ConversionReport)",
            "export_qir": "QdAlgorithmIR -> bytes + ConversionReport",
            "capability": "check_conversion('qir', 'qd_algorithm_ir')",
        }
    )
    deferred_branch: str = "feature/qir-adapter"
    acceptance_criteria: List[str] = field(
        default_factory=lambda: [
            "Lossless or Approximate kind is explicit per profile",
            "Unsupported constructs raise ConversionError with report",
            "No production dependency on proprietary closed compilers without extras",
        ]
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_qir_contract() -> Dict[str, Any]:
    return QirAdapterContract().to_dict()
