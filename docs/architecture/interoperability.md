# Interoperability Architecture

最終更新: 2026-11-25  
関連: DEC-011, DEC-020–026、`docs/competitive-comparison.md`

## 位置づけ

Qudas は SDK を置き換えない。Gate / Annealing / Classical の Domain IR を Hub とし、変換・Pipeline・Problem/Algorithm 再利用を提供する。

## Conversion Graph

- ノード: 形式 ID（例: `qiskit`, `qd_algorithm_ir`, `qubo`, `dimod_bqm`）
- エッジ: converter + `ConversionKind` + notes
- API: `register` / `can_convert` / `find_paths` / `convert` / report
- 経路 kind = 構成エッジの最弱クラス
- `ArtifactConverterRegistry` は Pipeline Artifact 専用（別責務）

## 意味保存

| Domain | 検証 |
|--------|------|
| Gate（ユニタリ） | 行列同値（全体位相許容） |
| Gate（測定あり） | 確率分布比較（ユニタリのみ適用しない） |
| QUBO / Ising | 係数・定数・エネルギー |
| Pipeline | 型・依存・演算 type_id |

## OpenQASM / QIR

| 機能 | 状態 |
|------|------|
| OpenQASM2 basic | 対応（Qiskit） |
| OpenQASM3 basic/param/measure | 部分対応（外部パーサ） |
| Classical control / dynamic | 未対応（Capability 明示） |
| QIR | Adapter Contract のみ |

## Pipeline Model Exchange

ONNX 参考: Envelope / schema_version / node type_id / validation。  
再実装しない: protobuf 完全互換、巨大独自演算子集合、分散ランタイム。
