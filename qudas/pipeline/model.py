"""永続化可能な Pipeline Model（実行時 QdPipeline とは分離）。

注意: QdPipeline.models は学習・実行オブジェクトの格納領域であり、
本モジュールの Pipeline Model（計算仕様）とは別概念である。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from qudas.model.schema import SCHEMA_VERSION, ModelType
from qudas.model.envelope import QdModelEnvelope

from .registry import ComponentRegistry, default_registry


@dataclass
class QdPipelineNode:
    """Pipeline 上の計算ノード。"""

    id: str
    type_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type_id": self.type_id,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QdPipelineNode":
        if "id" not in data or "type_id" not in data:
            raise ValueError("Node requires 'id' and 'type_id'")
        return cls(
            id=str(data["id"]),
            type_id=str(data["type_id"]),
            parameters=dict(data.get("parameters") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class QdPipelineEdge:
    """データ依存（src → dst）。"""

    source: str
    target: str
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"source": self.source, "target": self.target}
        if self.label is not None:
            d["label"] = self.label
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QdPipelineEdge":
        if "source" not in data or "target" not in data:
            raise ValueError("Edge requires 'source' and 'target'")
        return cls(
            source=str(data["source"]),
            target=str(data["target"]),
            label=data.get("label"),
        )


@dataclass
class ControlFlowSpec:
    """制御フロー仕様。

    kind:
      - sequential: ノード順次（edges のトポロジまたは nodes の順序）
      - loop: 明示的な反復（iterations / loop_over）
      - branch / join / parallel: 表現のみ（実行は未対応 → validate で警告）
    """

    kind: str = "sequential"
    iterations: int = 1
    loop_over: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "iterations": self.iterations,
            "loop_over": self.loop_over,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ControlFlowSpec":
        data = data or {}
        return cls(
            kind=str(data.get("kind", "sequential")),
            iterations=int(data.get("iterations", 1)),
            loop_over=data.get("loop_over"),
            details=dict(data.get("details") or {}),
        )


@dataclass
class QdPipelineModel:
    """再利用・保存可能な Pipeline 計算モデル。"""

    nodes: List[QdPipelineNode] = field(default_factory=list)
    edges: List[QdPipelineEdge] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    control_flow: ControlFlowSpec = field(default_factory=ControlFlowSpec)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = SCHEMA_VERSION
    name: str = "pipeline"

    # ---- 構築 ---------------------------------------------------------
    def add_node(
        self,
        node_id: str,
        type_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "QdPipelineModel":
        if any(n.id == node_id for n in self.nodes):
            raise ValueError(f"Duplicate node id: {node_id}")
        self.nodes.append(
            QdPipelineNode(
                id=node_id,
                type_id=type_id,
                parameters=dict(parameters or {}),
                metadata=dict(metadata or {}),
            )
        )
        return self

    def add_edge(
        self, source: str, target: str, label: Optional[str] = None
    ) -> "QdPipelineModel":
        ids = {n.id for n in self.nodes}
        if source not in ids or target not in ids:
            raise ValueError(f"Edge endpoints must exist: {source} -> {target}")
        self.edges.append(QdPipelineEdge(source=source, target=target, label=label))
        return self

    def with_parameters(self, **updates: Any) -> "QdPipelineModel":
        """パラメータの部分更新（コピーを返す）。"""
        clone = self.from_dict(self.to_dict())
        clone.parameters.update(updates)
        return clone

    def with_node_parameters(self, node_id: str, **updates: Any) -> "QdPipelineModel":
        clone = self.from_dict(self.to_dict())
        for n in clone.nodes:
            if n.id == node_id:
                n.parameters.update(updates)
                return clone
        raise KeyError(f"Unknown node id: {node_id}")

    # ---- 検証 ---------------------------------------------------------
    def validate(self, registry: Optional[ComponentRegistry] = None) -> List[str]:
        """妥当性検証。エラー文字列のリスト（空なら OK）。"""
        registry = registry or default_registry
        errors: List[str] = []
        if not self.nodes:
            errors.append("Pipeline model has no nodes")
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            errors.append("Node ids must be unique")
        id_set = set(ids)
        for e in self.edges:
            if e.source not in id_set:
                errors.append(f"Edge source missing: {e.source}")
            if e.target not in id_set:
                errors.append(f"Edge target missing: {e.target}")
        for n in self.nodes:
            if not registry.has(n.type_id):
                errors.append(f"Unregistered type_id: {n.type_id} (node={n.id})")
        kind = self.control_flow.kind
        if kind in ("branch", "join", "parallel"):
            errors.append(
                f"Control flow kind={kind!r} is schema-only; "
                "runtime execution is not implemented"
            )
        if self.control_flow.iterations < 1:
            errors.append("control_flow.iterations must be >= 1")
        # サイクル検出（edges がある場合）
        if self.edges:
            adj: Dict[str, List[str]] = {i: [] for i in id_set}
            for e in self.edges:
                if e.source in adj:
                    adj[e.source].append(e.target)
            visiting, visited = set(), set()

            def dfs(u: str) -> bool:
                if u in visiting:
                    return True
                if u in visited:
                    return False
                visiting.add(u)
                for v in adj.get(u, []):
                    if dfs(v):
                        return True
                visiting.remove(u)
                visited.add(u)
                return False

            for i in id_set:
                if dfs(i):
                    errors.append("Pipeline graph contains a cycle")
                    break
        return errors

    # ---- シリアライズ -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "parameters": self.parameters,
            "context": self.context,
            "control_flow": self.control_flow.to_dict(),
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QdPipelineModel":
        if not isinstance(data, dict):
            raise TypeError("Pipeline model data must be a dict")
        required = ("nodes",)
        for k in required:
            if k not in data:
                raise ValueError(f"Pipeline model missing required field: {k}")
        nodes = [QdPipelineNode.from_dict(n) for n in data.get("nodes") or []]
        edges = [QdPipelineEdge.from_dict(e) for e in data.get("edges") or []]
        return cls(
            nodes=nodes,
            edges=edges,
            parameters=dict(data.get("parameters") or {}),
            context=dict(data.get("context") or {}),
            control_flow=ControlFlowSpec.from_dict(data.get("control_flow")),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            metadata=dict(data.get("metadata") or {}),
            version=str(data.get("version", SCHEMA_VERSION)),
            name=str(data.get("name", "pipeline")),
        )

    def to_envelope(self) -> QdModelEnvelope:
        return QdModelEnvelope(
            model_type=ModelType.PIPELINE,
            payload=self.to_dict(),
            schema_version=self.version,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            parameters=self.parameters,
            metadata=self.metadata,
        )

    @classmethod
    def from_envelope(cls, envelope: QdModelEnvelope) -> "QdPipelineModel":
        if envelope.model_type != ModelType.PIPELINE:
            raise ValueError(
                f"Expected model_type=pipeline, got {envelope.model_type}"
            )
        return cls.from_dict(envelope.payload)

    def to_json(self, indent: int = 2) -> str:
        return self.to_envelope().to_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "QdPipelineModel":
        return cls.from_envelope(QdModelEnvelope.from_json(text))

    def save(self, path: Union[str, Path]) -> None:
        self.to_envelope().save(path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "QdPipelineModel":
        return cls.from_envelope(QdModelEnvelope.load(path))

    # ---- 実行接続 -----------------------------------------------------
    def ordered_node_ids(self) -> List[str]:
        """edges があればトポロジカル順、なければ定義順。"""
        if not self.edges:
            return [n.id for n in self.nodes]
        id_set = {n.id for n in self.nodes}
        indeg = {i: 0 for i in id_set}
        adj: Dict[str, List[str]] = {i: [] for i in id_set}
        for e in self.edges:
            adj[e.source].append(e.target)
            indeg[e.target] += 1
        queue = [i for i in [n.id for n in self.nodes] if indeg[i] == 0]
        order: List[str] = []
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        if len(order) != len(id_set):
            raise ValueError("Cannot order nodes (cycle or disconnected edges)")
        return order

    def materialize(
        self, registry: Optional[ComponentRegistry] = None
    ) -> "Any":
        """登録済みコンポーネントから実行用 QdPipeline を生成する。"""
        from .pipeline import QdPipeline

        registry = registry or default_registry
        errors = self.validate(registry)
        # branch/join/parallel は実行不能として raise
        hard = [e for e in errors if "schema-only" in e or "Unregistered" in e or "cycle" in e or "no nodes" in e]
        if hard:
            raise ValueError("Invalid pipeline model: " + "; ".join(hard))

        node_map = {n.id: n for n in self.nodes}
        steps = []
        for nid in self.ordered_node_ids():
            node = node_map[nid]
            # グローバル parameters をノードへマージ（ノード優先）
            params = dict(self.parameters)
            params.update(node.parameters)
            instance = registry.create(node.type_id, params)
            steps.append((nid, instance))

        pipe = QdPipeline(steps)
        if self.context:
            pipe.set_context(dict(self.context))
        # 明示ループ
        iterations = self.control_flow.iterations
        if iterations > 1:

            class _Loop:
                loop_num = iterations

            pipe.global_iterator = _Loop()
        return pipe

    def run(
        self,
        X: Any = None,
        y: Any = None,
        mode: str = "optimize",
        registry: Optional[ComponentRegistry] = None,
    ) -> Any:
        """materialize して fit/optimize/predict を実行。"""
        pipe = self.materialize(registry=registry)
        if mode == "fit":
            return pipe.fit(X, y)
        if mode == "optimize":
            return pipe.optimize(X, y)
        if mode == "predict":
            return pipe.predict(X)
        raise ValueError(f"Unknown mode: {mode}")

    # ---- 可視化用（QPLAT） --------------------------------------------
    def visualization_graph(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {"id": n.id, "label": n.id, "type_id": n.type_id} for n in self.nodes
            ],
            "edges": [e.to_dict() for e in self.edges],
            "control_flow": self.control_flow.to_dict(),
        }
