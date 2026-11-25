"""代表的 AlgorithmSpec の登録と Pipeline Model 生成。"""

from __future__ import annotations

from qudas.pipeline.model import QdPipelineModel, ControlFlowSpec
from qudas.pipeline import builtins as _builtins  # noqa: F401 — Step 登録

from .base import AlgorithmSpec
from .registry import register_algorithm
from . import runners as _runners  # noqa: F401 — runner 登録


def _sequential_pipeline(name: str, type_id: str, params=None) -> QdPipelineModel:
    model = QdPipelineModel(name=name, control_flow=ControlFlowSpec(kind="sequential"))
    model.add_node("solve", type_id, parameters=params or {})
    return model


CLASSICAL_EXACT = register_algorithm(
    AlgorithmSpec(
        algorithm_id="classical.exact",
        name="Classical Exact Enumeration",
        version="0.1.0",
        description="Brute-force enumeration for small binary problems (n<=20).",
        parameters={},
        compatible_problems=["maxcut", "knapsack"],
        required_dependencies=[],
        backend_requirements=["classical"],
        pipeline_model=_sequential_pipeline(
            "classical_exact", "qudas.algorithm.classical_exact"
        ),
        paper={},
    )
)

ANNEALING_DIMOD = register_algorithm(
    AlgorithmSpec(
        algorithm_id="annealing.dimod_exact",
        name="Annealing via dimod ExactSolver",
        version="0.1.0",
        description="QUBO solve using dimod.ExactSolver through QdAnnealingExecutor.",
        parameters={},
        compatible_problems=["maxcut", "knapsack", "binary_optimization"],
        required_dependencies=["dimod"],
        backend_requirements=["annealing", "dimod"],
        pipeline_model=_sequential_pipeline(
            "annealing_dimod", "qudas.algorithm.annealing_dimod"
        ),
        paper={},
    )
)

QAOA_QISKIT = register_algorithm(
    AlgorithmSpec(
        algorithm_id="qaoa.qiskit",
        name="QAOA (Qiskit sample)",
        version="0.1.0",
        description=(
            "Sample QAOA for MaxCut using Qiskit Aer. "
            "This is NOT a complete reproduction of Farhi et al.; "
            "use for Import/Adapt demonstration."
        ),
        parameters={"p": 1, "shots": 512, "opt_steps": 20, "lr": 0.4},
        parameter_schema={
            "type": "object",
            "properties": {
                "p": {"type": "integer", "minimum": 1},
                "shots": {"type": "integer", "minimum": 1},
                "seed": {"type": ["integer", "null"]},
                "opt_steps": {"type": "integer", "minimum": 1},
                "lr": {"type": "number"},
            },
        },
        compatible_problems=["maxcut"],
        required_dependencies=["qiskit", "qiskit-aer"],
        backend_requirements=["gate", "qiskit"],
        pipeline_model=_sequential_pipeline(
            "qaoa_qiskit",
            "qudas.algorithm.qaoa_qiskit",
            params={"p": 1, "shots": 512, "opt_steps": 20, "lr": 0.4},
        ),
        paper={
            "title": "A Quantum Approximate Optimization Algorithm",
            "authors": "Farhi, Goldstone, Gutmann",
            "arXiv": "1411.4028",
            "note": "Sample adaptation only; not a bit-exact reproduction.",
        },
    )
)
