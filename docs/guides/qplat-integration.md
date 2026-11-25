# QPLAT Integration Contract

今回 **QPLAT 本体の改修は行いません**。Qudas は独立ライブラリのままです。

接続は `qudas.qplat.QplatContract`（Optional Adapter）経由の宣言的 dict/JSON を想定します。

## 提供 API

| 用途 | メソッド |
|------|----------|
| Problem 一覧 | `list_problems()` |
| Algorithm 一覧 | `list_algorithms()` |
| 入出力 / パラメータ Schema | `get_algorithm_schema(id)` |
| Pipeline JSON | `pipeline_to_json(model)` |
| 可視化用グラフ | `pipeline_visualization(model)` |
| パラメータ取得・変更 | `get_pipeline_parameters` / `set_pipeline_parameters` |
| Result JSON | `result_to_json` |
| 比較 | `compare_to_json` |
| 変換 Capability | `list_conversions` |

## 依存方針

- Qudas → QPLAT: **依存しない**
- QPLAT Core → Qudas: **ハード依存しない**（Adapter / IPC / サブプロセス等で dict 契約を消費）

## 今後 QPLAT 側で必要な作業

1. `QplatContract` の JSON を GUI フォーム生成にバインド
2. Pipeline 可視化コンポーネント（nodes/edges）
3. 実験 Result のグラフ表示（timing / resources は方式別に分離）
4. 実行ジョブのキューイング（有料 QPU は明示オプトイン）
