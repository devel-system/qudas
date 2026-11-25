"""Advanced Pipeline Runtime。

QdPipeline（Stable 逐次オーケストレータ）とは分離し、
QdPipelineModel の Branch / Join / Sub-Pipeline / Parallel / Loop を解釈する。

分散 Scheduler は実装しない。Parallel は同一プロセス内の論理並列
（任意で ThreadPool）に限定する。
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from qudas.pipeline.registry import ComponentRegistry, default_registry


class PipelineRuntimeError(RuntimeError):
    """Advanced runtime 実行時エラー。"""


@dataclass
class RuntimeState:
    """実行中状態（永続化しない）。"""

    values: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    branch_choices: Dict[str, str] = field(default_factory=dict)
    iteration: int = 0


def _eval_condition(condition: Dict[str, Any], state: RuntimeState, value: Any) -> bool:
    """単純な条件評価（任意コード実行は禁止）。

    サポート:
      {"path": "context.flag"} -> bool(context['flag'])
      {"path": "value"} -> bool(value)
      {"path": "context.x", "equals": ...}
      {"path": "context.x", "op": "lt"|"le"|"gt"|"ge"|"eq"|"ne", "value": ...}
    """
    path = condition.get("path", "value")
    if path == "value":
        current = value
    elif path.startswith("context."):
        key = path.split(".", 1)[1]
        current = state.context.get(key)
    elif path.startswith("results."):
        key = path.split(".", 1)[1]
        current = state.results.get(key)
    else:
        current = state.values.get(path, value)

    if "equals" in condition:
        return current == condition["equals"]
    op = condition.get("op")
    if op is None:
        return bool(current)
    rhs = condition.get("value")
    if op == "eq":
        return current == rhs
    if op == "ne":
        return current != rhs
    if op == "lt":
        return current < rhs
    if op == "le":
        return current <= rhs
    if op == "gt":
        return current > rhs
    if op == "ge":
        return current >= rhs
    raise PipelineRuntimeError(f"Unsupported condition op: {op}")


class AdvancedPipelineRuntime:
    """QdPipelineModel 向け高度実行エンジン。"""

    SUPPORTED_KINDS = {
        "sequential",
        "loop",
        "branch",
        "join",
        "parallel",
        "graph",
    }

    def __init__(
        self,
        model: Any,
        registry: Optional[ComponentRegistry] = None,
        *,
        parallel_workers: int = 0,
        max_iterations: Optional[int] = None,
    ) -> None:
        self.model = model
        self.registry = registry or default_registry
        self.parallel_workers = int(parallel_workers)
        # Loop 安全性: model.control_flow.iterations と max_iterations の小さい方
        cf = model.control_flow
        declared = int(getattr(cf, "iterations", 1) or 1)
        hard_cap = int(
            max_iterations
            if max_iterations is not None
            else (cf.details or {}).get("max_iterations", max(declared, 1))
        )
        if hard_cap < 1:
            raise ValueError("max_iterations must be >= 1")
        self.max_iterations = hard_cap
        self.iterations = min(declared, hard_cap) if cf.kind == "loop" else 1
        if cf.kind == "loop" and declared > hard_cap:
            # 明示ループでも安全上限を適用
            self.iterations = hard_cap

    def validate(self) -> List[str]:
        errors: List[str] = []
        kind = self.model.control_flow.kind
        if kind not in self.SUPPORTED_KINDS:
            errors.append(f"Unsupported control_flow.kind={kind!r}")
        errors.extend(self.model.validate(self.registry))
        # model.validate が古い schema-only を返す場合に備えフィルタ
        errors = [e for e in errors if "schema-only" not in e]
        return errors

    def run(
        self,
        X: Any = None,
        y: Any = None,
        mode: str = "optimize",
    ) -> Any:
        errs = self.validate()
        hard = [
            e
            for e in errs
            if "Unregistered" in e
            or "cycle" in e
            or "no nodes" in e
            or "Unsupported control_flow" in e
        ]
        if hard:
            raise PipelineRuntimeError("Invalid pipeline model: " + "; ".join(hard))

        state = RuntimeState(
            values={"__input__": X, "__y__": y},
            context=dict(self.model.context or {}),
        )
        if X is not None:
            state.context.setdefault("X", X)
        kind = self.model.control_flow.kind

        last: Any = X
        loops = self.iterations if kind == "loop" else 1
        for i in range(loops):
            state.iteration = i
            if kind == "branch":
                last = self._run_branch(state, X if i == 0 else last, y, mode)
            elif kind == "parallel":
                last = self._run_parallel(state, X if i == 0 else last, y, mode)
            elif kind in ("graph", "join", "sequential", "loop"):
                last = self._run_graph(state, X if i == 0 else last, y, mode)
            else:  # pragma: no cover
                raise PipelineRuntimeError(f"Unhandled kind: {kind}")
            # loop 終了条件（任意）
            term = (self.model.control_flow.details or {}).get("terminate_when")
            if term and _eval_condition(term, state, last):
                break
        state.results["__final__"] = last
        return last

    # ---- helpers -----------------------------------------------------
    def _node_map(self):
        return {n.id: n for n in self.model.nodes}

    def _adj(self) -> Dict[str, List[Tuple[str, Optional[str]]]]:
        adj: Dict[str, List[Tuple[str, Optional[str]]]] = {n.id: [] for n in self.model.nodes}
        for e in self.model.edges:
            adj[e.source].append((e.target, e.label))
        return adj

    def _predecessors(self) -> Dict[str, List[Tuple[str, Optional[str]]]]:
        pred: Dict[str, List[Tuple[str, Optional[str]]]] = {
            n.id: [] for n in self.model.nodes
        }
        for e in self.model.edges:
            pred[e.target].append((e.source, e.label))
        return pred

    def _create_instance(self, node) -> Any:
        params = dict(self.model.parameters or {})
        params.update(node.parameters or {})
        # 認証情報キーを拒否
        for banned in ("token", "api_key", "password", "secret"):
            if banned in params:
                raise PipelineRuntimeError(
                    f"Refusing to materialize secrets in node {node.id}: {banned}"
                )
        if node.type_id == "qudas.subpipeline":
            return params  # 特殊扱い
        return self.registry.create(node.type_id, params)

    def _execute_node(
        self,
        node,
        value: Any,
        y: Any,
        mode: str,
        state: RuntimeState,
    ) -> Any:
        meta = node.metadata or {}
        role = meta.get("role")

        if node.type_id == "qudas.subpipeline" or role == "subpipeline":
            return self._run_subpipeline(node, value, y, mode, state)

        if role == "branch":
            # 条件ノード: 値はそのまま通し、選択を記録
            cond = meta.get("condition") or (self.model.control_flow.details or {}).get(
                "condition"
            )
            if not cond:
                raise PipelineRuntimeError(f"Branch node {node.id} missing condition")
            choice = "true" if _eval_condition(cond, state, value) else "false"
            state.branch_choices[node.id] = choice
            # 後続ノードへは元の値を渡す（choice は branch_choices に保持）
            state.results[node.id] = value
            state.values[f"{node.id}.__choice__"] = choice
            return value

        if role == "join":
            # 既に並列ランタイムから list/dict が渡されている場合はそのまま整形
            strategy = meta.get("join", "dict")
            if isinstance(value, list) and strategy in ("list", "dict", "last"):
                out = value if strategy != "last" else (value[-1] if value else None)
                if strategy == "dict":
                    out = {str(i): v for i, v in enumerate(value)}
                state.results[node.id] = out
                return out
            pred = self._predecessors().get(node.id, [])
            merged = {
                src: state.results.get(src, state.values.get(src))
                for src, _ in pred
            }
            if strategy == "list":
                out = [merged[s] for s, _ in pred]
            elif strategy == "last":
                out = value
            else:
                out = merged
            state.results[node.id] = out
            return out

        inst = self._create_instance(node)
        # context 共有
        if hasattr(inst, "set_context"):
            inst.set_context(state.context)
        elif hasattr(inst, "global_params"):
            inst.global_params = state.context

        out = value
        if mode == "fit" and hasattr(inst, "fit"):
            inst.fit(value, y)
            out = inst
        elif mode == "predict" and hasattr(inst, "predict"):
            out = inst.predict(value)
        elif hasattr(inst, "optimize"):
            out = inst.optimize(value, y)
        elif hasattr(inst, "transform"):
            out = inst.transform(value)
        elif callable(inst):
            out = inst(value)
        else:
            raise PipelineRuntimeError(
                f"Component {node.type_id} has no fit/optimize/predict/transform"
            )

        state.results[node.id] = out
        state.values[node.id] = out
        # context へ結果を反映（オプション）
        if meta.get("store_as"):
            state.context[meta["store_as"]] = out
        return out

    def _run_subpipeline(self, node, value, y, mode, state: RuntimeState):
        params = dict(self.model.parameters or {})
        params.update(node.parameters or {})
        nested = params.get("model") or params.get("pipeline_model")
        if nested is None:
            raise PipelineRuntimeError(f"Sub-pipeline node {node.id} requires parameters.model")
        from qudas.pipeline.model import QdPipelineModel

        if isinstance(nested, QdPipelineModel):
            sub_model = nested
        elif isinstance(nested, dict):
            # envelope or raw
            if "payload" in nested and nested.get("model_type") == "pipeline":
                sub_model = QdPipelineModel.from_dict(nested["payload"])
            elif "model_type" in nested:
                from qudas.model.envelope import QdModelEnvelope

                sub_model = QdPipelineModel.from_envelope(QdModelEnvelope.from_dict(nested))
            else:
                sub_model = QdPipelineModel.from_dict(nested)
        else:
            raise PipelineRuntimeError("Sub-pipeline model must be dict or QdPipelineModel")

        # 親 context を継承（秘密は既に拒否）
        sub_model = QdPipelineModel.from_dict(sub_model.to_dict())
        sub_model.context = {**state.context, **(sub_model.context or {})}
        rt = AdvancedPipelineRuntime(
            sub_model,
            registry=self.registry,
            parallel_workers=self.parallel_workers,
        )
        out = rt.run(value, y, mode=mode)
        state.results[node.id] = out
        return out

    def _run_graph(self, state: RuntimeState, X: Any, y: Any, mode: str) -> Any:
        """DAG 実行（Branch ラベル縁を考慮）。"""
        if not self.model.edges:
            last = X
            for n in self.model.nodes:
                last = self._execute_node(n, last, y, mode, state)
            return last

        node_map = self._node_map()
        adj = self._adj()
        pred = self._predecessors()
        indeg = {nid: 0 for nid in node_map}
        for e in self.model.edges:
            indeg[e.target] += 1

        # 入口
        ready = [nid for nid, d in indeg.items() if d == 0]
        done: Set[str] = set()
        skipped: Set[str] = set()
        last = X
        incoming_values: Dict[str, Any] = {nid: X for nid in ready}

        while ready:
            # parallel kind では ready をまとめて実行
            batch = list(ready)
            ready = []
            if self.model.control_flow.kind == "parallel" and len(batch) > 1:
                results_batch = self._parallel_map(
                    batch, node_map, incoming_values, y, mode, state
                )
            else:
                results_batch = {}
                for nid in batch:
                    node = node_map[nid]
                    inval = incoming_values.get(nid, last)
                    results_batch[nid] = self._execute_node(node, inval, y, mode, state)

            for nid, out in results_batch.items():
                done.add(nid)
                last = out
                # 後続
                for tgt, label in adj.get(nid, []):
                    meta = node_map[nid].metadata or {}
                    # branch 選択でラベルフィルタ
                    if nid in state.branch_choices:
                        choice = state.branch_choices[nid]
                        if label is not None and label != choice:
                            # この縁は選ばれない → 下流をスキップ候補
                            self._mark_unreachable(tgt, adj, skipped, done)
                            continue
                    elif meta.get("role") == "branch" and label is not None:
                        choice = state.branch_choices.get(nid)
                        if choice and label != choice:
                            self._mark_unreachable(tgt, adj, skipped, done)
                            continue

                    if tgt in skipped:
                        continue
                    # join: 全前任が done or skipped になるまで待つ
                    preds = pred[tgt]
                    if not preds:
                        continue
                    unresolved = [
                        p
                        for p, _ in preds
                        if p not in done and p not in skipped
                    ]
                    if unresolved:
                        # まだ待つ
                        incoming_values[tgt] = out
                        continue
                    # アクティブな前任からの値
                    active = [
                        (p, lbl)
                        for p, lbl in preds
                        if p in done and p not in skipped
                    ]
                    if not active:
                        skipped.add(tgt)
                        continue
                    if len(active) == 1:
                        incoming_values[tgt] = state.results.get(active[0][0], out)
                    else:
                        incoming_values[tgt] = {
                            p: state.results.get(p) for p, _ in active
                        }
                    if tgt not in done and tgt not in ready:
                        ready.append(tgt)

        return last

    def _mark_unreachable(self, start: str, adj, skipped: Set[str], done: Set[str]):
        stack = [start]
        while stack:
            u = stack.pop()
            if u in skipped or u in done:
                continue
            skipped.add(u)
            for v, _ in adj.get(u, []):
                stack.append(v)

    def _parallel_map(self, batch, node_map, incoming_values, y, mode, state):
        workers = self.parallel_workers
        if workers <= 0:
            # 論理並列: 逐次だが独立入力で実行（共有 state はメインスレッド）
            # 並列性を主張しつつ安全性優先
            out = {}
            # 共有状態への書き込み競合を避けるため、ノードローカルに実行し後でマージ
            local_states = []
            for nid in batch:
                st = RuntimeState(
                    values=dict(state.values),
                    context=dict(state.context),
                    results=dict(state.results),
                    branch_choices=dict(state.branch_choices),
                    iteration=state.iteration,
                )
                node = node_map[nid]
                inval = incoming_values.get(nid)
                result = self._execute_node(node, inval, y, mode, st)
                local_states.append((nid, result, st))
            for nid, result, st in local_states:
                state.results[nid] = st.results.get(nid, result)
                state.values[nid] = result
                out[nid] = result
            return out

        # ThreadPool: 共有 state は使わず完了後にマージ（コンテキストはコピー）
        def job(nid):
            st = RuntimeState(
                values=dict(state.values),
                context=copy.deepcopy(state.context),
                results=dict(state.results),
                iteration=state.iteration,
            )
            node = node_map[nid]
            inval = incoming_values.get(nid)
            result = self._execute_node(node, inval, y, mode, st)
            return nid, result, st.results.get(nid, result)

        out = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(job, nid) for nid in batch]
            for fut in as_completed(futs):
                nid, result, stored = fut.result()
                state.results[nid] = stored
                state.values[nid] = result
                out[nid] = result
        return out

    def _run_branch(self, state: RuntimeState, X: Any, y: Any, mode: str) -> Any:
        """control_flow.kind=branch 専用: details で true/false パスを指定可能。"""
        details = self.model.control_flow.details or {}
        # グラフに role=branch があるなら graph 実行
        if any((n.metadata or {}).get("role") == "branch" for n in self.model.nodes):
            return self._run_graph(state, X, y, mode)

        cond = details.get("condition") or {"path": "value"}
        choice = "true" if _eval_condition(cond, state, X) else "false"
        state.branch_choices["__root__"] = choice
        true_nodes = details.get("true_nodes") or []
        false_nodes = details.get("false_nodes") or []
        selected = true_nodes if choice == "true" else false_nodes
        node_map = self._node_map()
        last = X
        if selected:
            for nid in selected:
                last = self._execute_node(node_map[nid], last, y, mode, state)
        else:
            # ラベル付き edges にフォールバック
            return self._run_graph(state, X, y, mode)
        # join ノード
        join_id = details.get("join")
        if join_id:
            last = self._execute_node(node_map[join_id], last, y, mode, state)
        return last

    def _run_parallel(self, state: RuntimeState, X: Any, y: Any, mode: str) -> Any:
        details = self.model.control_flow.details or {}
        branches = details.get("branches")
        if not branches:
            return self._run_graph(state, X, y, mode)
        node_map = self._node_map()
        # 各 branch は node id リスト
        branch_inputs = {i: X for i in range(len(branches))}
        results = {}

        def run_branch(idx, node_ids):
            st = RuntimeState(
                values=dict(state.values),
                context=copy.deepcopy(state.context),
                results={},
                iteration=state.iteration,
            )
            last = X
            for nid in node_ids:
                last = self._execute_node(node_map[nid], last, y, mode, st)
            return idx, last, st.results

        if self.parallel_workers > 0:
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as ex:
                futs = [
                    ex.submit(run_branch, i, br) for i, br in enumerate(branches)
                ]
                for fut in as_completed(futs):
                    idx, last, res = fut.result()
                    results[idx] = last
                    state.results.update({f"branch{idx}.{k}": v for k, v in res.items()})
        else:
            for i, br in enumerate(branches):
                idx, last, res = run_branch(i, br)
                results[idx] = last
                state.results.update({f"branch{idx}.{k}": v for k, v in res.items()})

        join_id = details.get("join")
        merged = [results[i] for i in range(len(branches))]
        if join_id:
            return self._execute_node(node_map[join_id], merged, y, mode, state)
        return merged
