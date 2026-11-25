# Conversion Graph

関連: DEC-023、`docs/architecture/interoperability.md`

## 概要

Qudas の形式変換を、関数の散在ではなく **グラフ** として管理します。

- ノード: `qiskit`, `openqasm2`, `qd_algorithm_ir`, `qubo`, `ising`, ...
- エッジ: converter + `ConversionKind` + availability

## API

```python
from qudas.model import ConversionGraph, get_default_graph
from qudas.exchange import ensure_graph_initialized, convert, check_conversion

ensure_graph_initialized()
print(check_conversion("qubo", "ising"))
obj, report = convert(qubo_dict, "qubo", "ising")
```

## 経路 kind

`weakest_kind`: unsupported > approximate > problem_level > lossless

## 非目標

- Gate↔Annealing 任意変換
- ArtifactConverterRegistry の置換（Pipeline Artifact 専用として維持）
