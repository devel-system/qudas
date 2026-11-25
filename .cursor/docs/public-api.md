# Public API

最終更新: 2026-09-18 / 版: **0.2.6**  
ラベルは Phase 7 で確定。**Stable を壊す変更は禁止**（0.2.x）。Candidate / Deprecated の扱いも下記ポリシーに従う。

## ラベル定義

| Label | 意味 | 互換 |
|-------|------|------|
| **Stable** | 第一級の利用者向け契約。破壊変更禁止 | 0.2.x で維持。削除・シグネチャ破壊はメジャー区切り（目標 v0.3）で計画 |
| **Candidate** | 利用可・有用だが安定宣言前。拡張者向け含む | 予告付きで変更しうる。トップレベル未 export のものが多い |
| **Deprecated** | 残存互換のみ。新規利用非推奨 | 警告あり。削除目標は v0.3（正確な版はリリース計画） |
| **Internal** | 公開契約外 | 予告なく変更しうる |

抽出元: `qudas/__init__.py` および各サブパッケージ `__init__.py`、Accepted DEC-010〜017、tests / examples。

---

## Stable

### トップレベル（`from qudas import ...`）

| Symbol | 役割 |
|--------|------|
| `QuData` | **変換専用ユーザ向け第一級入口**。`input()` / `output()` は annealing IR/Output へ委譲 |
| `QdAnnealingIR` | 変換の **canonical 実装**（`from_*` / `to_*` / 演算 / `.qubo` / `.prob`） |
| `QdAnnealingInput`, `QdAnnealingOutput`, `QdAnnealingExecutor`, `QdAnnealingBlock` | Annealing 実行スタック |
| `QdGateInput`, `QdGateOutput`, `QdGateExecutor`, `QdGateIR`, `QdAlgorithmIR`, `QdGateBlock` | Gate 実行スタック |
| `QdPipeline` | オーケストレータ（`fit` / `optimize` / `predict` / `get_results` / `get_models` / context 系） |
| `QdPipelineModel` | **永続化可能な Pipeline 仕様**（Additive。`QdPipeline.models` とは別） |
| `__version__` | パッケージ版文字列 |

### サブパッケージ（Candidate → 利用推奨）

| Import | 役割 |
|--------|------|
| `qudas.problem` | MaxCut / Knapsack / BinaryOptimization |
| `qudas.algorithm` | AlgorithmSpec / AlgorithmPackage / 登録 Solver |
| `qudas.api` | `solve` / `compare` / `catalog` |
| `qudas.result` | `QdExperimentResult` / `compare_results` |
| `qudas.model` | Envelope / ConversionReport |
| `qudas.exchange` | Import / Port / Capability |
| `qudas.qplat` | QPLAT 向け宣言的 Contract |

### 変換専用ワークフロー（Stable）

```python
from qudas import QuData
# Executor 不要
ir = QuData.input().from_array(arr)
bqm = ir.to_dimod_bqm()
```

同等に Stable: `from qudas import QdAnnealingIR` の直接利用。

`QuData.input(mode=...)` は **`mode="annealing"` のみ Stable**。  
`mode="gate"` は未サポート（`NotImplementedError`）— Stable に含めない。

### 実行ワークフロー（Stable・変換とは別経路）

```python
from qudas import QdAnnealingBlock, QdAnnealingInput, QdAnnealingExecutor
```

### サブパッケージから常用（Stable）

| Import | Symbol |
|--------|--------|
| `qudas.pipeline` | `QdPipeline` |
| `qudas.gate` | 上記 Gate クラス（正式名） |
| `qudas.annealing` | 上記 Annealing クラス（正式名） |
| `qudas.qudata` | `QuData` のみ（レガシーは Deprecated） |

### Pipeline サポート範囲（Stable 契約）

公式にサポートする API: `fit` / `optimize` / `predict` / `get_results` / `get_models` / context 系。  
**未サポート（Stable 外）:** 公開 `transform` / `fit_transform`、`set_params(step__param=...)` ネスト解決。

### Output 契約メモ（Stable）

`QuData.output()` → `QdAnnealingOutput`。結果は `results[block]["solution"|"energy"]` 形式。  
レガシー風の `.qubo` / 単一 `.result` 辞書は **現行 Output の Stable 契約ではない**。

---

## Candidate

利用あり／拡張点として重要だが、トップレベル未 export または別名のため、0.2.x での破壊は避けるが **安定宣言はしない**。

| Symbol / 領域 | Evidence / 備考 |
|---------------|-----------------|
| `qudas.core`: `QdInputBase` / `QdOutputBase` / `QdExecutorBase` | 拡張点。新名が正式。旧名 `QdInBase` 等は互換エイリアス |
| `qudas.core.statistics` | v0.2.1〜。output から利用 |
| `qudas.pipeline.steps`: `QdBaseStep` / `IteratorMixin` / `OptimizerMixin` | ステップ実装の入口。`BaseStep` は旧別名 |
| `qudas.pipeline`: `Pipeline`（=`QdPipeline`） | 互換別名 |
| Gate/Annealing 短縮別名（`QdGateIn`, `QdAnnIR`, `QdBlock` 等） | サブパッケージ互換。新規コードは正式名を推奨 |
| `pipeline.artifacts` / `converter` / `blocks.*` | examples/vqe 等で使用。トップレベル未 export |

---

## Deprecated

| Item | 移行先 | 削除目標 |
|------|--------|----------|
| `QuDataInput` / `QuDataOutput` | `QuData.input()` / `QuData.output()` または `QdAnnealingIR` / `QdAnnealingOutput` | v0.3（DeprecationWarning 済み） |
| 旧 `AnnealingExecutor`（CHANGELOG 記載） | `QdAnnealingExecutor` | 現行ツリーに無し（履歴のみ） |

変換 **能力** 自体は Deprecated ではない（Stable の `QuData` / `QdAnnealingIR` が担う）。

---

## Internal

- Executor 内部の `_run_*` / `_block_to_*` および provider 解決ヘルパ
- `QuDataBase`（薄いコンテナ）
- Amplify バージョン差分ヘルパ
- モジュール私有名（`_*`）全般

`pipeline.artifacts` / `converter` / `blocks` は examples 利用があるため **Candidate**（上記）。内部実装詳細の変更は許容。

---

## `__version__`

- 開発ツリー: リポジトリ直下 `VERSION` を優先（現在 **0.2.6**）
- インストール環境: `importlib.metadata.version("qudas")`
- どちらも無い場合: `0.0.0+unknown`
- トップレベル `__all__` に含む（Stable）

---

## Import surface

`import qudas` は gate / annealing / qudata / pipeline を **eager import** する（当面維持・仕様）。  
optional SDK 自体は遅延 import。

---

## 変更ポリシー（0.2.x）

1. **Stable を壊さない**（変換 annealing 経路・実行スタック・Pipeline 公式メソッド）
2. Candidate の縮小・リネームは予告可能だが、無断の大規模削除はしない
3. Deprecated へ新規機能を足さない
4. extras: **0.2.x は `annealing` 維持 + 変換最小依存を文書明示**。v0.3 で `annealing-convert` を Additive 追加予定（`annealing` からパッケージを削らない）

---

## Planned（v0.3 — 未実装）

| Item | 予定契約 |
|------|----------|
| `QdExecutorBase(..., fallback="raise"\|"naive")` | デフォルト `"raise"`（失敗は元例外を再送出）。`"naive"` は明示 opt-in（warning 維持）。`run` 引数には載せない |
| `QdGateExecutor` / `QdAnnealingExecutor` | 同上をコンストラクタで受け取る |
| Pipeline / `QuantumBlock` | 必要なら `executor` インスタンスまたは `executor_kwargs` を Additive 追加（fit 系 Stable は不変） |

0.2.x の現行挙動: SDK 失敗時は **warning + 暗黙 naive**（`fallback` 引数なし）。
