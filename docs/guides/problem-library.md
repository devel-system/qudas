# Problem Library

## 方針

Amplify 等と問題クラス数で競争しない。共通契約の上で **1 問題を複数 Algorithm で再利用**する。

## 共通契約（`OptimizationProblem`）

Variables / Domain / Objective / Constraints / Conversion / Decode / Evaluate / Feasibility

実装の正は `BinaryOptimizationProblem` + Domain IR（`QdAnnealingIR`）。QuData は変換 facade（DEC-011）。

## 実装済み

| Problem | 状態 |
|---------|------|
| MaxCut | 実装 |
| Knapsack | 実装（スラック付きペナルティ） |
| TSP | 実装（one-hot、小規模 brute force） |

## 複数 Algorithm

```python
from qudas.api import solve, compare
from qudas.problem import MaxCutProblem, TSPProblem

p = MaxCutProblem.from_edge_list([(0, 1), (1, 2), (0, 2)])
compare(p, algorithms=["classical.exact", "annealing.dimod_exact"])

tsp = TSPProblem(distances=[[0, 1, 1], [1, 0, 1], [1, 1, 0]])
solve(tsp, "classical.exact")
```

## 候補（未実装）

Graph Coloring / Portfolio / Scheduling — 必要性が明確なときのみ追加。
