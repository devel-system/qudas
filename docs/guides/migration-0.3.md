# Migration Guide（0.2.x → 次世代拡張）

本拡張は **後方互換の Additive 変更** です。既存コードの変更は不要です。

## そのまま動くもの

- `from qudas import QuData, QdPipeline, QdAnnealingIR, ...`
- `QdPipeline.fit` / `optimize` / `predict` / `models` / `results`
- Gate / Annealing Executor
- `qudas[gate]` / `qudas[annealing]` extras

## 新しい推奨パス（任意）

```python
from qudas.problem import MaxCutProblem
from qudas.api import solve, compare
from qudas import QdPipelineModel
```

## 用語の注意

| 用語 | 意味 |
|------|------|
| `QdPipeline.models` | 学習・実行オブジェクトの格納（既存） |
| `QdPipelineModel` | 永続化可能な Pipeline **仕様**（新規） |

## 追加された主な API（Additive）

- `qudas.model.ConversionGraph` / `qudas.exchange.convert`
- `qudas.pipeline.AdvancedPipelineRuntime`
- `qudas.problem.TSPProblem`
- `qudas.exchange.verify_*` / `get_qir_contract`

## 破壊的変更

なし（本ブランチ時点）。OpenQASM3 の classical control、QIR 実装、Gate↔Annealing 直変換、分散 Parallel は意図的に未提供です。

`QdPipelineModel.validate()` の `branch`/`join`/`parallel` に対する schema-only エラーは撤廃され、実行可能になりました（Additive）。
