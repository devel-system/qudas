# Competitive Comparison — Qudas vs Existing Tools

調査日: 2026-11-25  
ブランチ: `feature/next-gen-architecture`  
証拠区分: **実測** / **実装確認** / **公式ドキュメント確認** / **推測（未使用）**

本比較は Qudas の位置づけ（独立 Python ライブラリとしての SDK 横断変換・ハイブリッド Pipeline・交換可能な計算モデル）を維持したうえで、**再実装すべきでない機能**と**独自価値のある機能**を切り分けるために行う。

---

## 1. 比較対象と情報源

| ツール | 主な情報源 | 証拠 |
|--------|------------|------|
| Fixstars Amplify SDK v1 | [features](https://amplify.fixstars.com/en/docs/amplify/v1/features.html), [intermediate](https://amplify.fixstars.com/en/docs/amplify/v1/intermediate.html) | 公式ドキュメント確認 |
| D-Wave Ocean / dwave-hybrid | [flow](https://hybrid.readthedocs.io/en/latest/reference/flow.html), [conversion](https://hybrid.readthedocs.io/en/latest/reference/conversion.html) | 公式ドキュメント確認 |
| Qiskit | Qiskit API / OpenQASM 経由 | 実装確認（本リポジトリの依存） |
| NVIDIA CUDA-Q | cudaq-convert README 等 | 公式・第三者ドキュメント確認 |
| PennyLane | [from_qasm3](https://docs.pennylane.ai/en/latest/code/api/pennylane.from_qasm3.html), importing workflows | 公式ドキュメント確認 |
| scikit-learn Pipeline | [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) | 公式ドキュメント確認 |
| Covalent / Prefect | Prefect task mapping / serializers docs | 公式ドキュメント確認 |
| ONNX | [IR](https://onnx.ai/onnx/repo-docs/IR.html), [Versioning](https://onnx.ai/onnx/repo-docs/Versioning.html) | 公式ドキュメント確認 |

参考: Guppy / Qrisp / TKET / QIR — 回路表現・中間表現の観点のみ参照。独自コンパイラ再実装は非目標。

---

## 2. 観点別比較（要約）

| 観点 | Amplify | dwave-hybrid | Qiskit/PennyLane/CUDA-Q | sklearn | Prefect/Covalent | ONNX | Qudas（CURRENT→本開発） |
|------|---------|--------------|-------------------------|---------|------------------|------|-------------------------|
| Gate / Annealing / Classical 横断 | Annealing 特化 | Annealing+古典 hybrid | Gate 中心 | Classical | Classical 汎用 | Classical NN | **横断が目的** |
| データ形式変換 | 高度（次数低減・変数変換） | dimod BQM 中心 | 回路 import | — | — | protobuf graph | QuData / Domain IR + Conversion Graph |
| SDK 互換 | Amplify エコシステム | Ocean | 各 SDK 内 + QASM | — | — | ランタイム互換 | **Hub（複数 Domain）** |
| Pipeline 構築 | Solver 呼び出し中心 | Branch/Loop/Parallel 豊富 | 回路/微分 Pipeline | 逐次のみ | 分散 DAG | 計算グラフ | Runtime + **永続 Model** |
| Branch/Join/Parallel/Loop | 限定的 | **強い** | 限定 | 弱い | **強い（分散）** | 静的グラフ | 逐次+loop → **Branch/Join/Sub/論理 Parallel** |
| モデル保存・移植 | モデルオブジェクト中心 | Runnable は pickle 寄り | 回路/QASM | joblib | cloudpickle/JSON | **protobuf + opset** | **JSON Envelope + Registry** |
| Problem/Algorithm 再利用 | 問題定式化が強み | Sampler 合成 | Algorithm はアプリ側 | Estimator | Task | Operator | Problem / Algorithm Package |
| 再現性 | Solver 依存 | ハードウェア依存 | シミュレータ依存 | 高い | 実行環境依存 | 高い | 意味・構造の再現を優先 |
| 開発・保守コスト | 製品 SDK | Ocean スタック | 巨大 | 低い | インフラ重い | 標準は重い | **薄い Hub** を維持 |

---

## 3. 機能分類（実装判断）

### 3.1 Qudas 独自に実装すべき機能

| 機能 | 根拠 |
|------|------|
| Domain 横断の Conversion Graph（Lossless/Approximate/Problem-level/Unsupported） | 各 SDK は自領域内変換に強く、Gate↔Annealing↔Classical の意味クラス付き経路管理は Hub 固有 |
| `QdPipelineModel`（Runtime と分離した JSON 計算仕様） | sklearn は pickle、Prefect は実行オーケストレーション、hybrid は Runnable。量子ハイブリッドの**意味付き**永続化は薄い |
| Problem → 複数 Algorithm/Backend → 共通 Result 再評価 | Amplify/Ocean は強いが、Qudas の「問題とアルゴリズム分離 + 交換」は独自 |
| Capability / Conversion Report | 変換可否と意味保存の事前宣言は交換ライブラリの核心 |

### 3.2 既存 SDK を利用すべき機能

| 機能 | 委譲先 |
|------|--------|
| 制約定式化・高次低減・埋め込み | Amplify / Ocean |
| 量子コンパイル・最適化パス | Qiskit / TKET / CUDA-Q |
| Exact / Heuristic QUBO sampling | dimod / Amplify AE（ローカル） |
| OpenQASM3 パース（高度） | Qiskit qasm3 / openqasm3 / PennyLane |
| 分散スケジューリング | Prefect / Covalent（再実装しない） |

### 3.3 Adapter で接続すべき機能

| 機能 | 方針 |
|------|------|
| Qiskit Circuit ↔ Gate IR | 既存 `from_qasm` / `to_qiskit` + Capability |
| OpenQASM2/3 | Qiskit 経由。未対応構文は Capability で明示 |
| QIR | Adapter Contract のみ（独自 LLVM 禁止） |
| Amplify / dimod / PuLP / PyQUBO | 既存 `QdAnnealingIR` / QuData |
| CUDA-Q / PennyLane | OpenQASM 経由の間接接続を優先（直接依存は extras 検討） |

### 3.4 現段階では不要な機能

| 機能 | 理由 |
|------|------|
| 独自量子コンパイラ | 既存 SDK が優勢・保守不能 |
| Prefect 級分散実行 | インフラ責務が別。論理 Parallel で十分 |
| ONNX 完全互換 protobuf | 量子ハイブリッドの演算子集合が不一致。概念のみ参考 |
| Amplify 級の問題クラス網羅 | 数の競争ではなく再利用設計を優先 |
| 論文プログラム完全自動移植 | P2 は既存 Import/Package で実証する範囲 |

---

## 4. 優位性・劣位（推測禁止・根拠付き）

### Qudas が優位になりうる点（設計・実装確認）

- Gate / Annealing / Classical を**単一 Hub**で扱い、変換の意味クラスを明示できる（DEC-021）。
- Runtime `QdPipeline` と永続 `QdPipelineModel` を分離し、任意コード実行なしで復元できる（DEC-020）。
- Problem / Algorithm / Result / Exchange が分離され、同じ問題を複数アルゴリズムで比較できる（DEC-022）。

### 既存ツールが優位な点（公式ドキュメント確認）

- **Amplify**: 制約・次数低減・変数変換・Solver 向け自動変換が成熟。
- **dwave-hybrid**: Branch / Parallel / Loop / Racing の実行合成が豊富。
- **Prefect**: 大規模並列・動的タスク・分散ランタイム。
- **ONNX**: Operator set / IR version / ランタイム検証のエコシステム。
- **Qiskit/PennyLane**: 回路・微分・ハードウェア実行の深さ。

### 同等または部分的

- sklearn 的な逐次 Pipeline API（Qudas は fit/optimize/predict。完全互換は非目標・DEC-015）。
- OpenQASM2 回路交換（各 SDK も対応）。

---

## 5. Benchmark 方針（コード比較は別ドキュメント）

実測比較は `docs/benchmarks/` および `benchmarks/` に記録する。

- **比較A Amplify**: 問題定義コード量・QUBO 係数一致・実行可能性（ローカル機能優先、有料 Solver 非必須）
- **比較B dwave-hybrid**: Loop/Branch/Parallel のデータフローと保存・再実行（機能比較中心）
- **比較C sklearn/Prefect**: API 簡潔さ・量子意味の保持
- **比較D ONNX**: Schema / Versioning / Validation の概念対応表

測定値・実装確認・推測を結果表で区別する。

---

## 6. 本開発への反映

1. Conversion Graph を `qudas.model` / `qudas.exchange` に追加し、Capability 一覧と統合（重複 Registry を増やさない）。
2. Pipeline Runtime に Branch/Join/Sub-Pipeline/論理 Parallel を Additive 追加。分散 Scheduler は作らない。
3. OpenQASM3 は外部パーサ経由の部分対応 + Capability 明示。QIR は Contract のみ。
4. Problem は TSP を追加し、MaxCut/Knapsack の再利用パターンを検証。Amplify 問題数競争はしない。
5. ONNX は Envelope / type_id version / validation の参考に留め、独自演算子爆発を避ける。
