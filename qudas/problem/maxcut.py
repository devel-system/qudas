"""MaxCut 問題。

カット重み C(x) = Σ_{(u,v)∈E} w_uv (x_u + x_v - 2 x_u x_v) を **最大化**。
QUBO 最小化エネルギーは H = -C(x)（定数項は含めない／offset は別管理）。

変数順序: ノードをソートした順で x{node}。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .binary import BinaryOptimizationProblem
from .registry import register_problem


def _edge_weight(data: dict) -> float:
    return float(data.get("weight", 1.0))


@register_problem("maxcut")
class MaxCutProblem(BinaryOptimizationProblem):
    """グラフの MaxCut。"""

    problem_id = "maxcut"
    sense = "maximize"

    def __init__(
        self,
        nodes: Sequence[Any],
        edges: Sequence[Tuple[Any, Any, float]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.nodes = list(nodes)
        self.edges = [(u, v, float(w)) for u, v, w in edges]
        variables = [f"x{n}" for n in self.nodes]
        qubo = self._build_qubo(self.nodes, self.edges)
        super().__init__(
            variables=variables,
            qubo=qubo,
            constraints=[],
            metadata=metadata or {},
            sense="maximize",
            constant=0.0,
        )

    @staticmethod
    def _build_qubo(
        nodes: Sequence[Any], edges: Sequence[Tuple[Any, Any, float]]
    ) -> Dict[Tuple[str, str], float]:
        """H = -Σ w (x_u + x_v - 2 x_u x_v) = Σ w (2 x_u x_v - x_u - x_v)."""
        qubo: Dict[Tuple[str, str], float] = {}

        def add(i: str, j: str, coeff: float) -> None:
            key = (i, j) if i <= j else (j, i)
            # 対角は (i,i) のまま
            if i == j:
                key = (i, i)
            qubo[key] = qubo.get(key, 0.0) + coeff

        for u, v, w in edges:
            xu, xv = f"x{u}", f"x{v}"
            add(xu, xu, -w)
            add(xv, xv, -w)
            # 二次項 2w x_u x_v。キーを正規化
            a, b = (xu, xv) if xu <= xv else (xv, xu)
            qubo[(a, b)] = qubo.get((a, b), 0.0) + 2.0 * w
        return qubo

    @classmethod
    def from_edge_list(
        cls,
        edges: Iterable[Tuple],
        nodes: Optional[Sequence[Any]] = None,
    ) -> "MaxCutProblem":
        parsed: List[Tuple[Any, Any, float]] = []
        node_set = set(nodes or [])
        for e in edges:
            if len(e) == 2:
                u, v = e[0], e[1]
                w = 1.0
            else:
                u, v, w = e[0], e[1], e[2]
            parsed.append((u, v, float(w)))
            node_set.add(u)
            node_set.add(v)
        ordered = list(nodes) if nodes is not None else sorted(node_set, key=lambda x: str(x))
        return cls(nodes=ordered, edges=parsed)

    @classmethod
    def from_networkx(cls, graph: Any) -> "MaxCutProblem":
        """NetworkX Graph（重み属性 weight 対応）。"""
        try:
            import networkx as nx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "networkx is required for from_networkx; pip install networkx"
            ) from exc
        nodes = list(graph.nodes())
        edges = []
        for u, v, data in graph.edges(data=True):
            edges.append((u, v, _edge_weight(data)))
        return cls(nodes=nodes, edges=edges, metadata={"source": "networkx"})

    def cut_value(self, assignment: Dict[str, int]) -> float:
        """カット重み和（最大化目的）。"""
        total = 0.0
        for u, v, w in self.edges:
            if int(assignment.get(f"x{u}", 0)) != int(assignment.get(f"x{v}", 0)):
                total += w
        return float(total)

    def evaluate(self, solution: Any) -> float:
        if not isinstance(solution, dict):
            raise TypeError("solution must be dict")
        return self.cut_value(solution)

    def classical_brute_force(self) -> Dict[str, Any]:
        """小規模厳密解（2^n）。n>20 は拒否。"""
        n = len(self.nodes)
        if n > 20:
            raise ValueError("classical_brute_force supports at most 20 nodes")
        best_val = float("-inf")
        best_sol: Dict[str, int] = {}
        for mask in range(1 << n):
            sol = {f"x{self.nodes[i]}": (mask >> i) & 1 for i in range(n)}
            val = self.cut_value(sol)
            if val > best_val:
                best_val = val
                best_sol = sol
        return {"solution": best_sol, "objective": best_val, "feasible": True}

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "nodes": list(self.nodes),
                "edges": [{"u": u, "v": v, "weight": w} for u, v, w in self.edges],
            }
        )
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaxCutProblem":
        if "nodes" in data and "edges" in data:
            edges = [(e["u"], e["v"], e.get("weight", 1.0)) for e in data["edges"]]
            return cls(nodes=data["nodes"], edges=edges, metadata=data.get("metadata"))
        raise ValueError("MaxCutProblem.from_dict requires nodes and edges")
