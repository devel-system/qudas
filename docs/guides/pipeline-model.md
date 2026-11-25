# Pipeline Model Guide

`QdPipeline` は **実行時オーケストレータ** です。`models` / `results` は学習・実行オブジェクトの共有領域であり、本ガイドの **Pipeline Model（計算仕様）** とは別物です。

## 概念

| 概念 | クラス | 役割 |
|------|--------|------|
| Runtime Pipeline | `QdPipeline` | fit / optimize / predict、反復、Block 連携 |
| Pipeline Model | `QdPipelineModel` | Node / Edge / Parameters / ControlFlow を持つ永続化可能モデル |
| Advanced Runtime | `AdvancedPipelineRuntime` | Branch / Join / Sub-Pipeline / Parallel / 安全 Loop |

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
- 認証情報（token / api_key 等）は Model に埋め込まない
- `dependencies` / `execution_requirements` で実行条件を宣言可能

## 制御フロー（実装済み範囲）

| kind | 状態 |
|------|------|
| sequential | 実装（既存 `QdPipeline`） |
| loop | 実装（`iterations` + `details.max_iterations` / `terminate_when`） |
| branch | 実装（`AdvancedPipelineRuntime`） |
| join | 実装 |
| parallel | 実装（論理並列。ThreadPool は任意） |
| graph | 実装（ラベル付き Branch 縁） |

分散 Scheduler / クラウド実行は非目標です（Prefect 等へ委譲）。

## Sub-Pipeline

```python
inner = QdPipelineModel(name="inner")
inner.add_node("a", "qudas.passthrough")
outer = QdPipelineModel(control_flow=ControlFlowSpec(kind="graph"))
outer.add_node("sub", "qudas.subpipeline", parameters={"model": inner.to_dict()})
```

## 既存 QdPipeline との関係

```python
pipe = model.materialize()  # sequential → QdPipeline / advanced → AdvancedPipelineRuntime
```

既存の `QdPipeline([...])` 手書き構築も引き続き利用できます。
