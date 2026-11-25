# Model Exchange Format

## Envelope（JSON）

ルートは `QdModelEnvelope`（schema_version / model_type / payload / IO schema /
parameters / backend_requirements / dependencies / metadata / provenance）。

`schema_version` 現在: **1.0.0**（メジャー不一致は読み込み拒否）。

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
| OpenQASM 3 | 未実装（外部コンパイラ前提） |
| QIR | Capability / Adapter 設計のみ |
| QUBO ↔ Ising / dimod / Amplify | 既存 `QdAnnealingIR` |

```python
from qudas.exchange import check_conversion, import_qiskit_circuit, port_annealing
from qudas.model import ConversionError

report = check_conversion("gate", "annealing")  # unsupported
```
