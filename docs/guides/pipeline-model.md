# Pipeline Model Guide

`QdPipeline` は **実行時オーケストレータ** です。`models` / `results` は学習・実行オブジェクトの共有領域であり、本ガイドの **Pipeline Model（計算仕様）** とは別物です。

## 概念

| 概念 | クラス | 役割 |
|------|--------|------|
| Runtime Pipeline | `QdPipeline` | fit / optimize / predict、反復、Block 連携 |
| Pipeline Model | `QdPipelineModel` | Node / Edge / Parameters / ControlFlow を持つ永続化可能モデル |

## 構築・保存・実行

```python
from qudas import QdPipelineModel
from qudas.problem import MaxCutProblem

model = QdPipelineModel(name="maxcut_classical")
model.add_node("solve", "qudas.algorithm.classical_exact")
model.save("pipeline.json")

loaded = QdPipelineModel.load("pipeline.json")
problem = MaxCutProblem.from_edge_list([(0, 1), (1, 2), (2, 0)])
results = loaded.run(problem, mode="optimize")
```

## シリアライズの制約

- JSON のみ（任意 Python コードは実行しない）
- 復元は `ComponentRegistry` に登録された `type_id` + パラメータ
- 未登録 Step は明示的にエラー
- `branch` / `join` / `parallel` はスキーマ表現のみ（実行未対応）

## 既存 QdPipeline との関係

```python
pipe = model.materialize()  # -> QdPipeline
pipe.optimize(problem)
```

既存の `QdPipeline([...])` 手書き構築も引き続き利用できます。
