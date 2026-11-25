# Algorithm Library Guide

Algorithm（宣言）と Pipeline Model（実装構造）は分離されています。

```python
from qudas.algorithm import get_algorithm, AlgorithmPackage
from qudas.problem import MaxCutProblem

spec = get_algorithm("qaoa.qiskit")
pkg = AlgorithmPackage.from_spec(spec)
pkg.save("qaoa_package.json")

problem = MaxCutProblem.from_edge_list([(0, 1), (1, 2), (2, 0)])
result = AlgorithmPackage.load("qaoa_package.json").run(
    problem, p=1, shots=256, opt_steps=10, seed=0
)
```

## 組み込み Algorithm

| ID | 内容 | 依存 |
|----|------|------|
| `classical.exact` | 全解列挙 (n≤20) | なし |
| `annealing.dimod_exact` | dimod ExactSolver | dimod |
| `qaoa.qiskit` | MaxCut 向け QAOA サンプル | qiskit, qiskit-aer |

`qaoa.qiskit` は Farhi et al. (arXiv:1411.4028) の **完全再現ではありません**。Import / Adapt 実証用です。

## Paper Algorithm Package フィールド

Algorithm ID / Name / Version / 論文情報 / Source Repo・Commit / License /
IO Schema / Parameters / Dependencies / Compatible Problems / Pipeline Model /
Tests / Reference Results
