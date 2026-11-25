# Decisions

調査日: 2026-09-18  
最終更新: 2026-09-18（DEC-010〜019 Accepted）

推測を Accepted にしない。コードから明らかな意図と、今後決める必要がある事項を分ける。

---

## A. Observed from code（事実に近い Accepted）

### DEC-001

**Title:** Core は backend SDK 非依存の抽象層とする  

**Status:** Accepted（CURRENT コードから確認）  

**Context:** Gate/Annealing が複数 backend を持ちうる。  

**Decision:** `qudas.core` は Input/Output/Executor ABC と statistics のみを持ち、qiskit/amplify/dimod 等を import しない。  

**Rationale:** 具象は domain モジュール側。  

**Consequences:** statistics は numpy に依存。matplotlib は output 側で遅延利用。

---

### DEC-002

**Title:** Gate / Annealing は対称的な Input–IR/Block–Executor–Output 構成  

**Status:** Accepted（CURRENT）  

**Context:** v0.2 で実行 API が追加された。  

**Decision:** 両ドメインとも `*Input` / `*Block` / `*IR` / `*Executor` / `*Output` を基本単位とする。Executor は `run`（と任意で `run_split`）を提供する。  

**Rationale:** CHANGELOG 0.2.0 および実装構造。  

**Consequences:** Pipeline は具象 Executor を内包せず委譲する。

---

### DEC-003

**Title:** Optional SDK は関数内遅延 import  

**Status:** Accepted（CURRENT）  

**Context:** extras で gate/annealing を分離している。  

**Decision:** SDK はモジュール先頭ではなく使用時に import する。  

**Rationale:** コアインストールでもモジュール構造を import 可能にする意図。  

**Consequences:** `import qudas` の eager re-export とは別問題（DEC-010）。欠落 SDK は実行時エラーまたは naive fallback（DEC-017）。

---

### DEC-004

**Title:** QUBO 変換ハブの実装本体は `QdAnnealingIR`  

**Status:** Accepted（CURRENT）  

**Context:** `QuData.input` docstring に「新IRを返却するラッパー。旧API互換」と明記。IR に旧 QuDataInput 移植コメント。  

**Decision:** 現行の変換実装の正は `QdAnnealingIR`。`QuDataInput`/`QuDataOutput` はレガシーとして残存。  

**Rationale:** コード上の明示。  

**Consequences:** Public 入口の位置づけは DEC-011 で「変換専用ユーザ向け facade」として再定義。テストは段階的に IR / facade へ寄せる（DEC-012）。

---

### DEC-005

**Title:** Pipeline はオーケストレータであり Executor 実装を持たない  

**Status:** Accepted（CURRENT）  

**Context:** `QuantumBlock` が `executor_cls` を受け取り実行する。  

**Decision:** Pipeline/Block は実行委譲点。backend 固有ロジックは Executor 側。  

**Rationale:** 実装構造。  

**Consequences:** artifacts/converter は Pipeline 拡張。責務重複を避ける。

---

### DEC-006

**Title:** パッケージバージョンの正は `VERSION` と `pyproject.toml`  

**Status:** Accepted（CURRENT メタデータ）  

**Context:** 複数箇所に版表記がある。  

**Decision:** リリース版の正は `VERSION` および `[project].version`（現在 **0.2.6**）。  

**Rationale:** packaging / CHANGELOG と一致。  

**Consequences:** README / Sphinx / docs の版表記を同期する。

---

## B. Phase 4 Accepted（2026-09-18 人間確認）

衝突対応: DEC-010↔C-007, DEC-011↔C-001/002/003, DEC-012↔C-006/014, DEC-013↔C-011, DEC-014↔C-010, DEC-015↔C-004/005, DEC-016↔C-008, DEC-017↔C-012。

### DEC-010

**Title:** トップレベル eager import 方針  

**Status:** Accepted  

**Context:** `qudas/__init__.py` が gate/annealing を常に import。関連: C-007。  

**Decision:** **当面は現状の eager re-export を維持し、仕様として文書化する。** 遅延化は v0.3 以降の検討事項とする。  

**Rationale:** 今すぐ変える利益より `from qudas import ...` 互換リスクが大きい。  

**Consequences:** 「コアだけ軽い import」は名乗らない。optional SDK は引き続き遅延（DEC-003）。

---

### DEC-011

**Title:** QuData の最終責務（Option A + 変換専用ユーザ配慮）  

**Status:** Accepted  

**Context:** ファサード / レガシー / AnnealingIR が並存。一方で **実行せず形式変換だけ使いたいユーザ** が存在する。  

**Decision:** **Option A を採用し、次のように役割を分ける。**

1. **実装の正（canonical）:** `QdAnnealingIR`（入力変換・演算）および対応する Output 側の現行クラス（`QdAnnealingOutput`）
2. **変換専用ユーザ向け Public 入口:** `QuData.input()` / `QuData.output()` を **第一級の変換 API** として維持する（単なる「消す予定の残骸」にしない）
3. **実行:** `QdAnnealingBlock` / `QdAnnealingInput` / `QdAnnealingExecutor` は変換とは別経路。変換のみの利用に Executor は不要
4. **ゲート:** `QuData.input(mode="gate")` は保証しない（削除または `NotImplementedError`。回路は `qudas.gate` を使う）

**Rationale:** 実装の単一化（AnnealingIR）と、変換だけしたいユーザの discoverability（`QuData`）を両立する。  

**Consequences:**

- README / public-api は「変換 = QuData → 実体 AnnealingIR」「実行 = annealing Executor」と明示する
- 新規 `from_*` / `to_*` / 演算は **AnnealingIR / AnnealingOutput にのみ追加**し、`QuData` は委譲に留める
- 変換に必要な SDK（pulp, dimod, …）と実行（amplify 等）の extras 分離は **DEC-018（Proposed）** で別途検討
- C-002（壊れた Output 例）は docs を現行 Output 契約に合わせて修正

---

### DEC-012

**Title:** レガシー `QuDataInput` / `QuDataOutput` の寿命  

**Status:** Accepted  

**Context:** 変換ロジックが `QdAnnealingIR` / `QdAnnealingOutput` と重複。ただし変換能力自体は維持必須（DEC-011）。  

**Decision:**

1. `QuDataInput` / `QuDataOutput` は **Deprecated**（実装の重複排除が目的）
2. **変換 Public API は廃止しない** — ユーザ移行先は:
   - 推奨: `QuData.input()` / `QuData.output()`（facade）
   - 直接: `QdAnnealingIR` / `QdAnnealingOutput`
3. 警告追加 → テストを IR/facade へ移行 → **v0.3 でレガシークラス削除**を目標とする（正確な版はリリース計画で確定）
4. Deprecated 期間中、レガシーへ新規機能を追加しない

**Rationale:** 二重実装のドリフトを止めつつ、変換専用ワークフローを残す。  

**Consequences:** `tests/qudata/*` の段階的移行。Sphinx の QuDataInput/Output ページに移行注記。

---

### DEC-013

**Title:** setuptools パッケージ列挙  

**Status:** Accepted  

**Context:** `packages = ["qudas"]` のみ。関連: C-011。  

**Decision:** **`find_packages()`（または同等の自動発見）へ変更する。**  

**Rationale:** サブパッケージ欠落リスクを下げる。  

**Consequences:** Phase 7 で `pyproject.toml` を修正し、wheel 内容を検証する。

---

### DEC-014

**Title:** インストール後の `__version__` 供給方法  

**Status:** Accepted  

**Context:** `VERSION` 非同梱時に `0.0.1` フォールバック。関連: C-010。  

**Decision:** **配布物（site-packages）では `importlib.metadata.version("qudas")` を用いる。開発ツリーではリポジトリ直下の `VERSION` を優先する。どちらも無い場合は `0.0.0+unknown`（旧 `0.0.1` フォールバックは廃止）。**  

**Rationale:** 配布物とメタデータの一致、およびローカル開発時の版の正確性。  

**Consequences:** Phase 7 相当で `qudas/__init__.py` を修正済み。pip インストール環境では親ディレクトリに `VERSION` が無いため metadata が使われる。

---

### DEC-015

**Title:** Pipeline sklearn 互換の到達度  

**Status:** Accepted  

**Context:** docstring が `transform` / `step__param` を示唆するが未実装。関連: C-004/C-005。  

**Decision:** **文書・契約を CURRENT に合わせて縮小する。** サポートする公式 API は `fit` / `optimize` / `predict` / `get_results` / `get_models` / context 系。`transform` 公開メソッドや `step__param` ネスト解決は **未サポート** と明記する（実装コミットしない）。`global_params` と context の整理は必要なら小さな follow-up とするが、sklearn 完全互換は目標にしない。  

**Rationale:** 実装コスト対効果が低い。オーケストレータとしての責務は既に足りる。  

**Consequences:** docstring / README / Sphinx の修正（Phase 8、必要なら Phase 6 で最小修正）。

---

### DEC-016

**Title:** core `__all__` 新名・旧名の正式化  

**Status:** Accepted  

**Context:** `__all__` に新名があるが未バインド。関連: C-008。  

**Decision:** **新名（`QdInputBase` / `QdOutputBase` / `QdExecutorBase`）を正式 export し、旧名は互換エイリアスとして維持する。**  

**Rationale:** `__all__` と実装を一致させ、拡張者向け API を明確にする。  

**Consequences:** Phase 6/7 で `qudas/core/__init__.py` の低リスク修正。

---

### DEC-017

**Title:** Fallback（naive 実行）の契約  

**Status:** Accepted  

**Context:** SDK 失敗時に naive 結果を返す。関連: C-012。  

**Decision:**

1. **移行期:** naive fallback 時に **warning** を出す（黙って成功させない）
2. **完成形:** デフォルトは失敗を **raise**。naive は明示オプション（例: `fallback="naive"`）でのみ許可
3. 完成形への切替タイミングはリリース計画で確定（目標: 次の互換ポリシー区切り、例 v0.3）

**Rationale:** デモ利便性と本番の決定性を分離する。  

**Consequences:** Phase 6 で warning 追加済み。オプション形状は **DEC-019**。デフォルト raise 切替は v0.3（WP-3.3）。

---

## C. Accepted follow-up（2026-09-18）

### DEC-018

**Title:** 変換用 extras と実行用 extras の分離  

**Status:** Accepted  

**Context:** DEC-011 の変換専用ユーザ。現状 `qudas[annealing]` は変換 SDK（dimod / PuLP / pandas / …）と Amplify 実行が同居。README も「分離は今後検討」と記載。  

**Options 比較:**

| Option | 内容 | 利点 | 欠点 |
|--------|------|------|------|
| A. 文書のみ | 変換最小依存を README 等に列挙。extras は現状維持 | 破壊なし・即時・CI 変更不要 | pip 一行では Amplify を避けられない |
| B. 0.2.x で extras 分割実装 | 例: `annealing-convert` 追加、`annealing` = convert + amplify | 変換専用のインストールが明確 | CI / docs / `all` 同期。Amplify は `to_amplify`/`from_amplify` でも必要で「完全分離」にならない |
| C. `annealing` から Amplify 除去 | 実行を別 extra へ | 変換ユーザが軽い | **破壊的**（既存 `qudas[annealing]` 利用者） |

**Decision:**

1. **0.2.x（現行）:** **Option A（文書明示）** を採用する。`qudas[annealing]` / `qudas[all]` の中身は変更しない。
2. **変換に必要な依存（文書上の最小セット）:**
   - コア済み: `numpy`, `matplotlib`
   - 形式変換 SDK: `dimod`, `PuLP`, `pyqubo`, `pandas`, `networkx`, `sympy`, `scipy`
   - **Amplify は変換の一部（`from_amplify` / `to_amplify`）と実行（Fixstars 等）の両方で使う。** 変換専用でも Amplify 形式が要る場合は Amplify の個別インストール、または従来どおり `qudas[annealing]`
3. **v0.3 目標（非破壊・Additive）:** Option B 相当で `annealing-convert`（上記変換 SDK、Amplify なし）を **追加**し、`annealing` = `annealing-convert` + `amplify` と定義し直す。既存 `annealing` 利用者は依存が増えない方向を維持（中身の意味は「変換+実行」のまま）。**`annealing` からパッケージを削る変更はしない。**

**Rationale:**

- Amplify が「実行専用」ではないため、今すぐの extras 分割は利用者期待と実装がずれやすい
- 変換専用ユーザの当面の痛みは「何が要るか分からない」であり、文書で大半は解消できる
- packaging の破壊変更を避け、Additive extras は互換ポリシー区切り（v0.3）で十分な準備のうえ行う

**Consequences:**

- README / 利用者向け docs に変換最小セットを書く（DEC 番号は出さない）
- `pyproject.toml` の extras は **0.2.6 では変更しない**
- WP-6 実装（extras 追加）は v0.3 バックログ。実装時は Architecture 再確認（CI の `.[annealing]` 参照）
- `public-api.md` の extras 注記を本 DEC に合わせて更新

---

### DEC-019

**Title:** Executor `fallback` オプションの API 形状（v0.3 実装前提）  

**Status:** Accepted（設計のみ。0.2.x では **未実装**）  

**Context:** DEC-017 完成形。現行 Gate/Annealing Executor は SDK 失敗時に warning + naive 結果。`QdExecutorBase` は `provider*` のみ。`QuantumBlock` は `executor_cls()` を引数なしで生成。WP-3.2。  

**Options 比較:**

| Option | 形状 | 判定 |
|--------|------|------|
| A. コンストラクタ `fallback=` のみ | `QdGateExecutor(fallback="naive")` | **採用** — `provider` と同階層、Pipeline でもインスタンスを渡せる |
| B. `run(..., fallback=)` のみ | 呼び出し毎 | 却下 — 呼び出し面が広がり、デフォルト契約が曖昧 |
| C. A+B（run が上書き） | 柔軟 | 見送り — v0.3 初回は単純さ優先。必要なら後続 |
| D. `provider_config["fallback"]` | 既存 dict に埋める | 却下 — 型・発見性が弱い |

**Decision（TARGET / v0.3）:**

1. **配置:** `QdExecutorBase.__init__` にキーワード専用引数 `fallback` を追加し、Gate / Annealing 具象も同じシグネチャで受けて `super()` に渡す。
2. **許容値:** `"raise"` | `"naive"` のみ。不正値は `__init__` で `ValueError`。
3. **デフォルト（v0.3）:** `"raise"` — SDK / provider 失敗時は **元例外を再送出**（新規 Public 例外クラスは作らない）。
4. **`"naive"`:** 現行と同じ naive 結果を返してよい。その場合も **warning は維持**（明示 opt-in でも黙って成功にしない）。
5. **`run` / `run_split`:** 引数に `fallback` を足さない（インスタンス設定のみ）。
6. **0.2.x:** 本 API は追加しない。現行（warning + 暗黙 naive）を維持。利用者向け docs では「v0.3 でデフォルト失敗化・明示 opt-in」と予告してよい（DEC 番号は出さない）。
7. **Pipeline / `QuantumBlock`（follow-up・必須ではない）:** 現状 `executor_cls()` のため `fallback` を渡せない。v0.3 では次のいずれか（または両方）を **Additive** で検討:
   - 生成済み `executor` インスタンスを受け取る
   - `executor_kwargs: dict` を `executor_cls(**kwargs)` に渡す  
   Pipeline 本体の Stable 契約（fit/optimize/predict）は変えない。
8. **Public ラベル:** 実装後は Executor コンストラクタの `fallback` を **Stable**（実行スタックの一部）。それまで docs 上は Planned。

**非目標:**

- provider 代替（例: amplify 失敗 → dimod）を `fallback` に載せない（それは `provider` / `provider_map`）
- ブロック単位の `fallback_map`（複雑化のため初回スコープ外）

**Rationale:** DEC-017 の「明示 opt-in」を最小の Public 面で実現する。基底への追加は対称な Gate/Annealing に一度で効く。`run` 引数化は避け、破壊的デフォルト変更は v0.3 に閉じる。

**Consequences:**

- 実装は WP-3.3（v0.3）。0.2.x production code は変更しない
- `public-api.md` に Planned 節を記載
- テスト計画: `fallback="raise"` で SDK 欠落時に例外、`fallback="naive"` で warning + 結果形
- 実装着手時に `QuantumBlock` 引数拡張が必要なら別タスク（Architecture 軽確認）

---

## D. Next-gen（2026-11-25, Additive）

### DEC-020

**Title:** Pipeline Runtime と Pipeline Model の分離  

**Status:** Accepted（実装済み・本ブランチ）  

**Context:** Pipeline を保存・交換・再利用可能な計算モデルにしたい。一方で既存 `QdPipeline.models` は学習オブジェクト格納域。  

**Decision:**
1. `QdPipeline` は実行オーケストレータとして維持（Stable 契約不変）
2. `QdPipelineModel` を新規追加し、Node/Edge/IO/Params/ControlFlow/Metadata/Version を JSON で永続化
3. 復元は Registry の `type_id` + params のみ。JSON からの任意コード実行は禁止
4. `branch`/`join`/`parallel` はスキーマのみ。実行は sequential + 明示 loop

**Consequences:** 用語衝突を docs で明示。Public に `QdPipelineModel` を Additive export。

### DEC-021

**Title:** Domain IR 非統合 + 変換クラス  

**Status:** Accepted  

**Decision:** Gate/Annealing/Classical/Problem を単一 IR に統合しない。共通は Envelope と ConversionReport（lossless / approximate / problem_level / unsupported）。Gate↔Annealing 任意変換は unsupported。

### DEC-022

**Title:** Problem / Algorithm / Result / QPLAT Adapter の配置  

**Status:** Accepted  

**Decision:** `qudas.problem` / `qudas.algorithm` / `qudas.result` / `qudas.exchange` / `qudas.qplat` を Additive 追加。Qudas↔QPLAT 相互ハード依存なし。Algorithm と Pipeline Model は分離し Package で束ねる。

---

## E. Superseded / Deprecated decisions

（旧 DEC ファイルなし。CHANGELOG の旧 `AnnealingExecutor` 削除予定は別履歴。）
