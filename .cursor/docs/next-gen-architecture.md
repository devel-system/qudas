# Next-Gen Architecture（調査・設計判断）

調査日: 2026-11-25 / ブランチ: `feature/next-gen-architecture`  
既存テスト: **85 passed, 1 skipped**（AMPLIFY_TOKEN 未設定）  
版: 現行 CURRENT = 0.2.6。本拡張は **Additive（後方互換）**。

---

## 1. 既存実装の調査結果（CURRENT）

### 再利用可能（そのまま活かす）

| 要素 | 判断 |
|------|------|
| `QdPipeline` | 実行オーケストレータ。`models`/`results`/`context`/`IteratorMixin` を維持。**Pipeline Model と混同しない** |
| `QdBaseEstimator` / `QdBaseStep` / Mixins | Step 実装の基底として継続利用 |
| `ArtifactBase` / Block / Converter | Block 連携を維持。Pipeline Model の Node 実行でも利用可 |
| `QdAnnealingIR` | QUBO canonical。Problem→モデル変換の正 |
| `QuData` facade | 変換専用入口として維持（DEC-011） |
| `QdGateIR` / `QdAlgorithmIR` | Gate Domain。`from_qasm` / `to_qiskit` を Import/Port の基盤に |
| Gate/Annealing Executor | Runtime。provider 解決を継続 |

### 拡張が必要

| 要素 | 理由 |
|------|------|
| Packaging / Public export | 新サブパッケージの公開 |
| Gate IR シリアライズ | Pipeline/Algorithm Package 用に `to_dict`/`from_dict` |
| OpenQASM3 / QIR | 既存は OpenQASM2 経由（Qiskit）。QIR は Adapter 設計のみ |

### 新規追加（必要最小）

| モジュール | 役割 |
|------------|------|
| `qudas.model` | 交換 Envelope / Schema Version / ConversionReport |
| `qudas.pipeline.model` | **永続化可能な Pipeline 仕様**（実行時 `QdPipeline` と分離） |
| `qudas.pipeline.registry` | Step/Block を識別子で復元（任意 Python コード実行禁止） |
| `qudas.problem` | Problem Library（MaxCut + Knapsack） |
| `qudas.algorithm` | Algorithm Spec / Package / 代表 Solver |
| `qudas.result` | 共通 Result・比較 |
| `qudas.exchange` | Import/Port/Capability |
| `qudas.qplat` | QPLAT 向け宣言的 Contract（本体非依存） |

---

## 2. 設計判断（TARGET → 本作業で実装）

### D1. Pipeline Runtime vs Pipeline Model の分離

- **`QdPipeline`**: 実行時オーケストレータ。`models` は学習・実行オブジェクト格納域（既存契約維持）。
- **`QdPipelineModel`**: 保存・交換・検証可能な計算モデル仕様（Node/Edge/IO/Params/ControlFlow/Metadata/Version）。
- 実行: `QdPipelineModel.materialize()` → `QdPipeline`（登録済みコンポーネントのみ）。

### D2. Domain IR は統合しない

Gate / Annealing / Classical / Problem は別 Domain。共通なのは Envelope・変換レポート・Result。

### D3. 変換の正当性クラス

`lossless` / `approximate` / `problem_level` / `unsupported`。Gate↔Annealing の任意変換は **unsupported**（問題経由のみ）。

### D4. JSON のみ・コード実行禁止

シリアライズは JSON。復元は Registry の `type_id` + params。未登録は明示エラー。

### D5. Algorithm と Pipeline Model の分離

Algorithm = メタデータ + 入出力契約 + 互換 Problem。実装構造は Pipeline Model として保持。

### D6. QPLAT

`qudas.qplat` は宣言的 JSON/dict を返す Optional Adapter。Qudas↔QPLAT の相互ハード依存なし。

### D7. 依存方向

```text
problem / algorithm / result / exchange / qplat
        → model, pipeline.model, annealing/gate (遅延), core
pipeline.model → pipeline.registry → pipeline.base（実行時 materialize のみ QdPipeline）
core は新規モジュールを知らない
```

### D8. 非目標（今回やらない）

分散実行エンジン / 独自 DSL / 論文自動変換 / QIR 完全実装 / QPLAT 本体改修 / Public API 破壊

---

## 3. E2E 対応マップ

| # | 条件 | 実装方針 |
|---|------|----------|
| 1 | MaxCut 定義 | `MaxCutProblem` |
| 2 | QAOA / Annealing / Classical | algorithm solvers + 既存 Executor |
| 3 | 目的関数で比較 | `QdExperimentResult` + `compare_results` |
| 4–5 | Pipeline 保存・再実行 | `QdPipelineModel` JSON |
| 6 | 問題変更で Algorithm 再利用 | Package.run(problem=...) |
| 7 | Qiskit 取り込み | `exchange.import_qiskit` / OpenQASM2 |
| 8 | Algorithm Package | `AlgorithmPackage` |
| 9 | 既存 API | 変更なし・回帰テスト |
| 10 | QPLAT 宣言形式 | `qplat.contract` |

---

## 4. 未対応として明示するもの（2026-11-25 更新）

- OpenQASM 3 完全: basic/param/measure は部分対応。classical control / dynamic は未対応
- QIR: Adapter Contract のみ（独自 LLVM なし）。実装は `feature/qir-adapter` 候補
- Gate→Annealing 任意変換: unsupported
- 分散 Parallel / クラウド Scheduler: 非目標（論理 Parallel のみ）
- Graph Coloring / Portfolio / Scheduling: 未実装（TSP で再利用性を追加検証済み）
- 論文プログラムの完全自動移植: 非目標（Package / Import による再利用実証）
