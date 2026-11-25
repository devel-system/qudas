# Model Exchange Format

## Envelope（JSON）

ルートは `QdModelEnvelope`（schema_version / model_type / payload / IO schema /
parameters / backend_requirements / dependencies / metadata / provenance）。

`schema_version` 現在: **1.0.0**（メジャー不一致は読み込み拒否）。

## Conversion Graph

形式をノード、変換をエッジとして経路探索します。

```python
from qudas.exchange import check_conversion, convert, ensure_graph_initialized

ensure_graph_initialized()
report = check_conversion("openqasm2", "qd_algorithm_ir")
ir, report = convert(qasm_str, "openqasm2", "qd_algorithm_ir")
```

経路全体の kind は構成エッジの最弱です。Approximate を含む経路を Lossless 扱いしません。

`ArtifactConverterRegistry` は Pipeline Artifact 専用で、本 Graph とは責務分離です。

## 変換クラス

| Kind | 意味 |
|------|------|
| lossless | 情報損失なし |
| approximate | 近似・サブセット |
| problem_level | 問題経由の翻訳（例: MaxCut→QUBO） |
| unsupported | 不可（明示エラー + ConversionReport） |

**Gate ↔ Annealing の任意変換は unsupported** です。問題レベルの翻訳（QAOA 等）を使います。

## 対応状況

| 経路 | 状態 |
|------|------|
| Qiskit ↔ QdAlgorithmIR | 実装（対応ゲートサブセット） |
| OpenQASM 2 → IR | 実装（Qiskit 経由） |
| OpenQASM 3 → IR | 部分対応（basic/param/measure）。classical control は拒否 |
| QIR | Adapter Contract のみ（`get_qir_contract()`） |
| QUBO ↔ Ising / dimod / Amplify | Conversion Graph + `QdAnnealingIR` |

## 意味保存検証

```python
from qudas.exchange import (
    verify_unitary_equivalence,
    verify_measurement_probabilities,
    verify_qubo_equivalence,
)
```

測定あり回路にはユニタリ比較を適用しません。

## OpenQASM3 Capability

`list_openqasm3_features()` / `openqasm3_capability_report()` で対応機能を明示します。
