import numpy as np
from qudas.pipeline.pipeline import QdPipeline
from qudas.pipeline.converter import ArtifactConverterRegistry
from qudas.pipeline.artifacts import QuantumArtifact, ClassicalArtifact

# Quantum Execution Step
from vqe_steps import (
    build_ansatz,
    VQEQuantumStep,
    VQEClassicalStep,
    zz_energy_from_counts,
)


# Converter function
def quantum_to_energy(artifact: QuantumArtifact) -> ClassicalArtifact:
    counts = artifact.data["counts"]
    energy = zz_energy_from_counts(counts)

    return ClassicalArtifact(
        data=energy,
        metadata={
            **artifact.metadata,
            "observable": "Z⊗Z",
        },
    )


# --- Dummy executor for example ---
class DummyExecutor:
    def run(self, q_input):
        # Replace with real executor
        return {"counts": {"00": 500, "11": 500}}


def main():
    theta = np.array([0.3, -0.2])

    context = {
        "params": {
            "theta": theta
        }
    }

    converter = ArtifactConverterRegistry()
    converter.register(QuantumArtifact, ClassicalArtifact, quantum_to_energy)
    executor = DummyExecutor()
    pipeline = QdPipeline(
        steps=[
            ("quantum", VQEQuantumStep(executor)),
            ("classical", VQEClassicalStep(lr=0.2)),
        ]
    )
    pipeline.set_context(context)

    print("=== VQE Example (Qudas) ===")
    print(f"Initial theta = {context['params']['theta']}")
    print("---------------------------")

    for epoch in range(10):
        # --- 現在のパラメータ ---
        theta = context["params"]["theta"]
        print(f"[Epoch {epoch}]")
        print(f"  Current theta : {theta}")

        # --- Ansatz 構築 ---
        block = build_ansatz(context["params"]["theta"])
        print(f"  Built ansatz  : {block}")

        # --- 量子実行 ---
        q_artifact = pipeline.named_steps["quantum"].optimize(block)
        print(f"  Quantum counts: {q_artifact.data['counts']}")

        # --- Convert Quantum -> Classical ---
        e_artifact = converter.convert(q_artifact, ClassicalArtifact)
        print(f"  EnergyArtifact: {e_artifact.data}")

        # --- 古典更新 ---
        classical = pipeline.named_steps["classical"]
        classical.set_context(pipeline.get_context())
        theta_artifact = classical.transform(e_artifact)

        print(f"  Updated theta : {theta_artifact.data}")
        print("---------------------------")

    print("=== VQE finished ===")
    print(f"Final theta = {context['params']['theta']}")

if __name__ == "__main__":
    main()