# Pipeline Runtime

## 二層構造

| 層 | 役割 |
|----|------|
| `QdPipeline` | Stable 逐次オーケストレータ（既存 API） |
| `AdvancedPipelineRuntime` | Branch/Join/Sub/Parallel/安全 Loop |

`QdPipelineModel.run()` が必要に応じて切り替える。

## Parallel の意味

- **論理並列**: 依存のないノードを独立入力で実行（既定）
- **ThreadPool**: `parallel_workers > 0` のときのみ。共有状態はコピー後マージ
- 分散 Scheduler は実装しない

## Loop 安全性

- `control_flow.iterations`
- `details.max_iterations`（上限）
- `details.terminate_when`（条件。任意コード不可）

## Portability

環境 A で `save` → 環境 B で `load` → Component 解決 → Backend 設定 → `run`。
測定ノイズの完全一致は要求しない。モデル意味と再現可能性を検証する。
