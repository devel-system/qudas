from qudas.annealing import QdAnnealingExecutor, QdAnnealingInput, QdAnnealingBlock
from qudas.annealing.output import QdAnnealingOutput

# === 簡単な QUBO ===
# minimize: x0 + x1 - 3*x0*x1
# qubo = {
#     ("x0", "x0"): 1,
#     ("x1", "x1"): 1,
#     ("x0", "x1"): -3,
# }

# minimize: x0 + x1 + x2 + x3 - 3*x0*x1 - x0*x2 - x0*x3 - x1*x2 - x1*x3 - x2*x3
qubo = {
    ("x0", "x0"): 1,
    ("x1", "x1"): 1,
    ("x2", "x2"): 2,
    ("x3", "x3"): 3,
    ("x0", "x1"): -3,
    ("x0", "x2"): 1,
    ("x0", "x3"): -1,
    ("x1", "x2"): 1,
    ("x1", "x3"): -1,
    ("x2", "x3"): -1,
}

# 入力作成
block = QdAnnealingBlock(qubo, label="block0")
anneal_input = QdAnnealingInput([block])

# 実行（dimod sampler）
executor = QdAnnealingExecutor(provider="dimod")
output: QdAnnealingOutput = executor.run(anneal_input)

# --- 出力確認 ---
print("=== Raw output ===")
print(output.results)

# --- 統計情報 ---
stats = output.results["block0"]["statistics"]
print("\n=== Statistics ===")
print("Energy std :", stats["energy"]["std"])
print("Unique bitstrings :", stats["bitstring"]["unique"])

# --- 可視化 ---
# energy の bar + std
output.visualize()