# Problem Library Guide

## 共通インターフェース

`OptimizationProblem` / `BinaryOptimizationProblem` が基盤です。

- 入力検証 / 目的関数評価 / 制約評価
- `to_model("qubo"|"ising"|"annealing_ir"|"dimod_bqm")`
- `decode` / `reevaluate`

## MaxCut

```python
from qudas.problem import MaxCutProblem
from qudas.api import solve, compare

problem = MaxCutProblem.from_edge_list([(0, 1), (1, 2), (2, 0)])
# または MaxCutProblem.from_networkx(G)

# カット最大化。QUBO は H = -C(x) を最小化
print(problem.classical_brute_force())
print(solve(problem, "annealing.dimod_exact").problem_objective)
```

符号・定数:

- 問題 sense = **maximize**（カット重み）
- QUBO energy = **-cut**（定数項なし）
- 変数名 = `x{node}`（ノード順はコンストラクタの `nodes`）

## Knapsack（共通設計の再利用検証）

`BinaryOptimizationProblem` + 容量制約（スラック二値展開 + ペナルティ）。

## 未実装問題の追加方法

TSP / Graph Coloring / Portfolio / Scheduling は未実装です。追加手順:

1. `BinaryOptimizationProblem` を継承（または制約付きで拡張）
2. `to_model("qubo")` と `evaluate` / `decode` を実装
3. `@register_problem("id")` で登録
4. 互換 Algorithm の `compatible_problems` を更新
