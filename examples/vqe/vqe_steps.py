import numpy as np
from qudas.pipeline.base import BaseStep, OptimizerStep
from qudas.gate.gate_ir import QdGateIR
from qudas.gate.block import QdGateBlock
from qudas.gate.input import QdGateInput
from qudas.pipeline.artifacts import QuantumArtifact, ClassicalArtifact


# Helper function to build a simple 2-qubit VQE ansatz
def build_ansatz(theta: np.ndarray) -> QdGateBlock:
    """Build a simple 2-qubit VQE ansatz."""
    gates = [
        QdGateIR("ry", targets=[0], params=[float(theta[0])]),
        QdGateIR("ry", targets=[1], params=[float(theta[1])]),
        QdGateIR("cx", targets=[1], controls=[0]),
    ]

    return QdGateBlock(gates=gates, num_qubits=2, label="vqe_ansatz")


# Helper function to compute <Z ⊗ Z> expectation value from counts
def zz_energy_from_counts(counts: dict) -> float:
    """Compute <Z ⊗ Z> expectation value."""
    shots = sum(counts.values())
    energy = 0.0

    for bitstring, count in counts.items():
        z0 = 1 if bitstring[-1] == "0" else -1
        z1 = 1 if bitstring[-2] == "0" else -1
        energy += z0 * z1 * count / shots

    return energy


# Quantum Execution Step
class VQEQuantumStep(OptimizerStep):
    """Quantum execution step."""

    def __init__(self, executor):
        self.executor = executor

    def optimize(self, X, y=None):
        q_input = QdGateInput(blocks=[X])
        result = self.executor.run(q_input)
        return QuantumArtifact(
            data=result,
            metadata={
                "shots": result.get("shots", None),
                "backend": result.get("backend", "dummy"),
            },
        )


# Classical Parameter Update Step
class VQEClassicalStep(BaseStep):
    """Classical parameter update step."""

    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def transform(self, artifact: ClassicalArtifact):
        energy = artifact.data

        ctx = self.get_context()
        theta = ctx["params"]["theta"]

        grad = 0.1 * np.ones_like(theta)
        new_theta = theta - self.lr * grad

        ctx["params"]["theta"] = new_theta

        return ClassicalArtifact(
            data=new_theta,
            metadata={"energy": energy},
        )

    def _dummy_grad(self, theta):
        return 0.1 * np.ones_like(theta)