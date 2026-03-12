from qudas.gate import QdGateBlock, QdGateIR, QdGateInput, QdGateExecutor
import matplotlib.pyplot as plt

# === 2量子ビットのGroverのアルゴリズムを実行 ===

# 重ね合わせ
superposition = QdGateBlock(
    label="superposition",
    gates=[
        QdGateIR(gate="h", targets=[0]),
        QdGateIR(gate="h", targets=[1]),
    ],
    num_qubits=2,
)

# オラクル
oracle = QdGateBlock(
    label="oracle",
    gates=[QdGateIR(gate="cz", targets=[0, 1])],
    num_qubits=2,
)

# 拡散
diffusion = QdGateBlock(
    label="diffusion",
    gates=[
        QdGateIR(gate="h", targets=[0, 1]),
        QdGateIR(gate="x", targets=[0, 1]),
        QdGateIR(gate="cz", targets=[0, 1]),
        QdGateIR(gate="x", targets=[0, 1]),
        QdGateIR(gate="h", targets=[0, 1]),
    ],
    num_qubits=2,
)

# アルゴリズム
grover_blocks = [
    superposition,
    oracle,
    diffusion,
]

# 実行
grover_input = QdGateInput(blocks=grover_blocks)

# # qiskitに変換
# ir = grover_input.to_ir()

# # qiskitに変換して回路を描画
# qiskit_circuit = ir.to_qiskit()
# qiskit_circuit.measure_all()
# qiskit_circuit.draw(output="mpl")
# plt.show()

# 実行
executor = QdGateExecutor(provider="qiskit", provider_config={"backend": "qiskit_simulator"})
output = executor.run(grover_input)
print(output.results)

# === Raw results ===
print("=== Raw results ===")
print(output.to_dict())

# === 統計情報 ===
stats = output.results["superposition"]["statistics"]
print("\n=== Statistics ===")
print("Probability std :", stats["probability"]["std"])
print("Unique bitstrings :", stats["bitstring"]["unique"])

# === 可視化 ===
# counts の histogram + std 表示
output.visualize()