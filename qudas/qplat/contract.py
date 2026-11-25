"""QPLAT GUI / 実験管理向けの宣言的インターフェース。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from qudas.problem.registry import list_problems
from qudas.algorithm.registry import list_algorithms, get_algorithm
from qudas.pipeline.model import QdPipelineModel
from qudas.result.common import QdExperimentResult
from qudas.result.compare import compare_results
from qudas.exchange.capabilities import list_capabilities


class QplatContract:
    """QPLAT が JSON で消費できる宣言データを返す。

    Qudas Core は本モジュールに依存しない。QPLAT Core も qudas に
    ハード依存せず、本 Adapter 経由の dict 契約を想定する。
    """

    @staticmethod
    def list_problems() -> List[Dict[str, Any]]:
        from qudas.problem import MaxCutProblem, KnapsackProblem

        items = list_problems()
        enriched = []
        for it in items:
            cls = {"maxcut": MaxCutProblem, "knapsack": KnapsackProblem}.get(
                it["problem_id"]
            )
            schema = {}
            if cls is not None:
                # 代表インスタンスなしでもスキーマを返す
                schema = {
                    "input_schema": {
                        "maxcut": {
                            "nodes": "array",
                            "edges": "array[{u,v,weight?}]",
                        },
                        "knapsack": {
                            "values": "array[number]",
                            "weights": "array[number]",
                            "capacity": "number",
                        },
                    }.get(it["problem_id"], {}),
                    "sense": getattr(cls, "sense", None),
                }
            enriched.append({**it, **schema})
        return enriched

    @staticmethod
    def list_algorithms() -> List[Dict[str, Any]]:
        out = []
        for row in list_algorithms():
            spec = get_algorithm(row["algorithm_id"])
            out.append(
                {
                    **row,
                    "parameters": spec.parameters,
                    "parameter_schema": spec.parameter_schema,
                    "input_schema": spec.input_schema,
                    "output_schema": spec.output_schema,
                    "required_dependencies": spec.required_dependencies,
                    "backend_requirements": spec.backend_requirements,
                    "paper": spec.paper,
                }
            )
        return out

    @staticmethod
    def get_algorithm_schema(algorithm_id: str) -> Dict[str, Any]:
        spec = get_algorithm(algorithm_id)
        return {
            "algorithm_id": spec.algorithm_id,
            "parameters": spec.parameters,
            "parameter_schema": spec.parameter_schema,
            "input_schema": spec.input_schema,
            "output_schema": spec.output_schema,
            "compatible_problems": spec.compatible_problems,
        }

    @staticmethod
    def pipeline_to_json(model: QdPipelineModel) -> Dict[str, Any]:
        return model.to_envelope().to_dict()

    @staticmethod
    def pipeline_visualization(model: QdPipelineModel) -> Dict[str, Any]:
        return model.visualization_graph()

    @staticmethod
    def get_pipeline_parameters(model: QdPipelineModel) -> Dict[str, Any]:
        return {
            "global": model.parameters,
            "nodes": {n.id: n.parameters for n in model.nodes},
        }

    @staticmethod
    def set_pipeline_parameters(
        model: QdPipelineModel, global_params: Optional[Dict] = None, node_params: Optional[Dict] = None
    ) -> QdPipelineModel:
        updated = model.from_dict(model.to_dict())
        if global_params:
            updated.parameters.update(global_params)
        if node_params:
            for nid, params in node_params.items():
                updated = updated.with_node_parameters(nid, **params)
        return updated

    @staticmethod
    def result_to_json(result: QdExperimentResult) -> Dict[str, Any]:
        return result.to_envelope().to_dict()

    @staticmethod
    def compare_to_json(results: List[QdExperimentResult]) -> Dict[str, Any]:
        return compare_results(results)

    @staticmethod
    def list_conversions() -> List[Dict[str, Any]]:
        return list_capabilities()
