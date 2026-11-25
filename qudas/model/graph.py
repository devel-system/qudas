"""Conversion Graph: 形式ノードと変換エッジによる経路探索。

ArtifactConverterRegistry（Pipeline Artifact 専用）とは責務を分離する。
既存 exchange.capabilities の宣言と同期して利用する。
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .conversion import ConversionError, ConversionReport
from .schema import ConversionKind


ConverterFn = Callable[[Any], Any]


_KIND_RANK = {
    ConversionKind.LOSSLESS: 0,
    ConversionKind.PROBLEM_LEVEL: 1,
    ConversionKind.APPROXIMATE: 2,
    ConversionKind.UNSUPPORTED: 3,
}


def weakest_kind(kinds: Iterable[ConversionKind]) -> ConversionKind:
    """経路全体の kind = 構成エッジの最弱。"""
    worst = ConversionKind.LOSSLESS
    for k in kinds:
        if isinstance(k, str):
            k = ConversionKind(k)
        if _KIND_RANK[k] > _KIND_RANK[worst]:
            worst = k
    return worst


@dataclass
class ConversionEdge:
    source: str
    target: str
    kind: ConversionKind
    converter: Optional[ConverterFn] = None
    notes: str = ""
    available: bool = True
    edge_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value if isinstance(self.kind, ConversionKind) else self.kind,
            "notes": self.notes,
            "available": self.available,
            "edge_id": self.edge_id,
            "has_converter": self.converter is not None,
        }


@dataclass
class ConversionPath:
    formats: List[str]
    edges: List[ConversionEdge]
    kind: ConversionKind

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formats": list(self.formats),
            "kind": self.kind.value,
            "edges": [e.to_dict() for e in self.edges],
            "hops": max(0, len(self.formats) - 1),
        }


@dataclass
class ConversionGraph:
    """形式間変換グラフ。"""

    _edges: Dict[Tuple[str, str], ConversionEdge] = field(default_factory=dict)
    _adj: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def register(
        self,
        source: str,
        target: str,
        kind: ConversionKind,
        converter: Optional[ConverterFn] = None,
        notes: str = "",
        available: bool = True,
        edge_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        source_l, target_l = source.lower(), target.lower()
        key = (source_l, target_l)
        if key in self._edges and not overwrite:
            raise ValueError(f"Conversion already registered: {source_l} -> {target_l}")
        if isinstance(kind, str):
            kind = ConversionKind(kind)
        edge = ConversionEdge(
            source=source_l,
            target=target_l,
            kind=kind,
            converter=converter,
            notes=notes,
            available=available,
            edge_id=edge_id or f"{source_l}->{target_l}",
        )
        if key not in self._edges:
            self._adj[source_l].append(target_l)
        self._edges[key] = edge

    def unregister(self, source: str, target: str) -> None:
        key = (source.lower(), target.lower())
        edge = self._edges.pop(key, None)
        if edge and edge.target in self._adj.get(edge.source, []):
            self._adj[edge.source] = [t for t in self._adj[edge.source] if t != edge.target]

    def has_edge(self, source: str, target: str) -> bool:
        return (source.lower(), target.lower()) in self._edges

    def get_edge(self, source: str, target: str) -> Optional[ConversionEdge]:
        return self._edges.get((source.lower(), target.lower()))

    def formats(self) -> List[str]:
        nodes = set()
        for s, t in self._edges:
            nodes.add(s)
            nodes.add(t)
        return sorted(nodes)

    def list_edges(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._edges.values()]

    def can_convert(self, source: str, target: str) -> ConversionReport:
        paths = self.find_paths(source, target, max_paths=1)
        if not paths:
            return ConversionReport.unsupported(
                source.lower(),
                target.lower(),
                "No conversion path in ConversionGraph",
            )
        path = paths[0]
        if path.kind == ConversionKind.UNSUPPORTED:
            return ConversionReport.unsupported(
                source.lower(),
                target.lower(),
                "Path contains unsupported edge",
                constraints=[e.notes for e in path.edges],
            )
        unavailable = [e for e in path.edges if not e.available]
        if unavailable:
            return ConversionReport.unsupported(
                source.lower(),
                target.lower(),
                "Path contains unavailable conversion",
                constraints=[e.notes for e in unavailable],
            )
        return ConversionReport(
            kind=path.kind,
            source=source.lower(),
            target=target.lower(),
            success=True,
            message=f"path={' -> '.join(path.formats)}",
            details=path.to_dict(),
        )

    def find_paths(
        self,
        source: str,
        target: str,
        *,
        max_hops: int = 6,
        max_paths: int = 8,
        allow_unavailable: bool = False,
    ) -> List[ConversionPath]:
        """BFS で短い経路を列挙。"""
        source_l, target_l = source.lower(), target.lower()
        if source_l == target_l:
            return [
                ConversionPath(
                    formats=[source_l],
                    edges=[],
                    kind=ConversionKind.LOSSLESS,
                )
            ]

        paths: List[ConversionPath] = []
        queue: deque = deque([(source_l, [source_l], [])])
        while queue and len(paths) < max_paths:
            node, node_path, edge_path = queue.popleft()
            if len(node_path) - 1 >= max_hops:
                continue
            for nxt in self._adj.get(node, []):
                if nxt in node_path:
                    continue
                edge = self._edges[(node, nxt)]
                if not allow_unavailable and (
                    not edge.available or edge.kind == ConversionKind.UNSUPPORTED
                ):
                    continue
                new_nodes = node_path + [nxt]
                new_edges = edge_path + [edge]
                if nxt == target_l:
                    paths.append(
                        ConversionPath(
                            formats=new_nodes,
                            edges=new_edges,
                            kind=weakest_kind(e.kind for e in new_edges),
                        )
                    )
                    if len(paths) >= max_paths:
                        break
                else:
                    queue.append((nxt, new_nodes, new_edges))
        # 短い経路・より lossless な経路を優先
        paths.sort(key=lambda p: (len(p.formats), _KIND_RANK[p.kind]))
        return paths

    def select_path(
        self,
        source: str,
        target: str,
        prefer: str = "shortest",
    ) -> Optional[ConversionPath]:
        paths = self.find_paths(source, target)
        if not paths:
            return None
        if prefer == "lossless":
            lossless = [p for p in paths if p.kind == ConversionKind.LOSSLESS]
            return lossless[0] if lossless else paths[0]
        return paths[0]

    def convert(
        self,
        obj: Any,
        source: str,
        target: str,
        *,
        prefer: str = "shortest",
    ) -> Tuple[Any, ConversionReport]:
        path = self.select_path(source, target, prefer=prefer)
        if path is None:
            report = ConversionReport.unsupported(
                source.lower(), target.lower(), "No conversion path"
            )
            raise ConversionError(report.message, report)
        if path.kind == ConversionKind.UNSUPPORTED:
            report = ConversionReport.unsupported(
                source.lower(), target.lower(), "Unsupported conversion path"
            )
            raise ConversionError(report.message, report)

        current = obj
        for edge in path.edges:
            if not edge.available:
                report = ConversionReport.unsupported(
                    edge.source, edge.target, edge.notes or "unavailable"
                )
                raise ConversionError(report.message, report)
            if edge.converter is None:
                report = ConversionReport(
                    kind=edge.kind,
                    source=edge.source,
                    target=edge.target,
                    success=False,
                    message=f"No callable converter for {edge.source} -> {edge.target}",
                )
                raise ConversionError(report.message, report)
            current = edge.converter(current)

        report = ConversionReport(
            kind=path.kind,
            source=source.lower(),
            target=target.lower(),
            success=True,
            message=f"Converted via {' -> '.join(path.formats)}",
            details=path.to_dict(),
        )
        return current, report


# プロセス共通グラフ（capabilities と同期）
default_conversion_graph = ConversionGraph()


def get_default_graph() -> ConversionGraph:
    return default_conversion_graph
