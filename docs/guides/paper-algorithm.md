# Paper Algorithm Import / Adapt Guide

三つの操作を区別します。

## 1. Import

既存 SDK 実装を Qudas へ取り込む。

```python
from qudas.exchange import import_qiskit_circuit, import_openqasm

ir, report = import_qiskit_circuit(qc)
ir2, report2 = import_openqasm(openqasm2_str)
```

任意の Python プロジェクト全体の自動解析は行いません。

## 2. Port

別 SDK / 形式へ変換。未対応は `ConversionError` + `ConversionReport`。

## 3. Adapt

アルゴリズム構造を別問題へ応用（例: QAOA の cost Hamiltonian を辺集合から再構成）。

```python
from qudas.exchange import adapt_qaoa_to_problem

result = adapt_qaoa_to_problem(maxcut_problem, parameters={"p": 1, "shots": 256})
```

単なる回路再利用とアルゴリズム構造再利用は別機能です。

## ライセンス

第三者コードのライセンスが不明な場合は再配布せず、`source_repository` / `source_commit` /
`license` の参照情報のみを Algorithm Package に記録してください。
