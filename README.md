
# Qudas (Quantum Data Transformation Library)

Qudasは、**量子・アニーリング・古典計算を横断して、実問題・アルゴリズム・Pipeline を保存・交換・実行・再利用する**ための Python ライブラリです。

主な柱:

1. **パイプライン（`QdPipeline`）** — 実行時オーケストレータ（`fit` / `optimize` / `predict`、`models` / `results` 共有）。詳細は 1-6。
2. **Pipeline Model（`QdPipelineModel`）** — Pipeline 全体を永続化・交換可能な **計算モデル仕様**（Node/Edge/Parameters）。`QdPipeline.models`（学習格納）とは別概念です。
3. **形式変換（`QuData`）** — QUBO まわりの変換専用 Public 入口。実装の正は `QdAnnealingIR`。回路はゲート用スタックで扱います。
4. **ゲート／アニーリング実行（オプション）** — `qudas[gate]` / `qudas[annealing]` の Input / IR / Executor / Output。
5. **Problem / Algorithm Library** — MaxCut / Knapsack / TSP 等の実問題と、Classical / Annealing / QAOA などの Algorithm Package。
6. **Conversion Graph / Model Exchange** — SDK 間の変換経路・意味保存検証・JSON Envelope。
7. **Advanced Pipeline Runtime** — Branch / Join / Sub-Pipeline / 論理 Parallel（分散 Scheduler は非目標）。

Sphinx API ドキュメントは [`sphinx_docs/source/`](sphinx_docs/source/)、ガイドは [`docs/guides/`](docs/guides/)、競合比較は [`docs/competitive-comparison.md`](docs/competitive-comparison.md) を参照してください。

本 README は

1. ライブラリ利用者向けドキュメント (Install / 全体像 / How-to)
2. 開発者向けドキュメント (開発フロー / コントリビュート手順)

の 2 つのセクションで構成されています。

---

## 1️⃣ ライブラリ利用者向けドキュメント

### 1-1. インストール
```bash
pip install qudas  # コアのみ（numpy + matplotlib）
pip install "qudas[annealing]"  # zsh 等では角括弧をクォート推奨
pip install "qudas[gate]"
pip install "qudas[all]"
# or
pip install git+https://github.com/devel-system/qudas.git@v0.2.6  # 開発版
```

**オプション**（zsh などでは角括弧をクォートしてください。上記コードブロック参照）
- `pip install qudas` … コアのみ（numpy + matplotlib）。`import qudas` はサブモジュールを eager load します（SDK は遅延 import）
- `pip install qudas[gate]` … 量子ゲート実行（Qiskit）
- `pip install qudas[annealing]` … データ変換 + アニーリング実行（dimod, Amplify, PyQUBO, PuLP, pandas, networkx, sympy, scipy）
- `pip install qudas[all]` … 上記 gate + annealing の全パッケージ

**変換だけ使う場合の依存（0.2.x）**

形式変換（`QuData` / `QdAnnealingIR`）に必要な SDK は現状 `annealing` extra にまとまっています。コア（numpy / matplotlib）に加え、変換で使うパッケージの目安は次のとおりです。

| 用途 | パッケージ |
|------|------------|
| 行列・表・グラフ・数式・古典最適化まわり | `dimod`, `PuLP`, `pyqubo`, `pandas`, `networkx`, `sympy`, `scipy` |
| Amplify の Poly 入出力（`from_amplify` / `to_amplify`）およびクラウド実行 | `amplify`（`qudas[annealing]` に含まれる） |

Amplify 形式やアニーリング実行が不要なら、上記の変換用パッケージだけを個別に入れても構いません。インストール一式が欲しければ従来どおり `qudas[annealing]` を使ってください。変換用 extra の分離は将来の非破壊追加として検討中です。

### 1-2. ライブラリの全体像（構成図）

v0.2.6 時点の概念図です（GitHub 上の README では下記パスの画像が表示されます）。

**QdPipeline** … ランタイムコンテキスト、古典ステップ列、Executor への委譲、最適化ループのフィードバックなどの関係。

![QdPipeline の概念（ランタイム・古典ステップ・量子／アニーリング Executor・最適化ループ）](sphinx_docs/source/img/fmqa_pipeline.png)

**変換と実行** … `QuData` / `QdAnnealingIR` が QUBO 系の相互変換、左がゲート実行スタック、右がアニーリング実行。変換のみの利用に Executor は不要です。

![QuData を中心としたゲート・アニーリングとデータの流れ](sphinx_docs/source/img/data_abstract.png)

### 1-3. クイックスタート
以下では代表的なユースケースを抜粋します。詳細は [examples/](examples/) および [docs/guides/](docs/guides/) も参照してください。

#### 1-3-0. Problem → Algorithm → 比較（新機能）
```python
from qudas.problem import MaxCutProblem
from qudas.api import solve, compare
from qudas.algorithm import AlgorithmPackage, get_algorithm

problem = MaxCutProblem.from_edge_list([(0, 1), (1, 2), (2, 0)])
print(solve(problem, "classical.exact").problem_objective)
print(solve(problem, "annealing.dimod_exact").problem_objective)

# Pipeline Model / Algorithm Package の保存と再利用
pkg = AlgorithmPackage.from_spec(get_algorithm("annealing.dimod_exact"))
pkg.save("annealing.json")
other = MaxCutProblem.from_edge_list([(0, 1), (1, 2)])
print(AlgorithmPackage.load("annealing.json").run(other).problem_objective)
```

比較サンプル: [`examples/maxcut_compare.py`](examples/maxcut_compare.py)

#### 1-3-1. QuData の生成
```python
from qudas import QuData

# QUBO (dict) から生成
qubo = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
qudata = QuData.input(qubo)
print(qudata.prob)               # => {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
```

#### 1-3-2. 四則演算
```python
q1 = QuData.input({('q0', 'q1'): 1.0})
q2 = QuData.input({('q0', 'q0'): 2.0})
print((q1 + q2).prob)            # => {('q0','q1'):1.0, ('q0','q0'):2.0}
print((q1 ** 2).prob)            # => {('q0','q1'):1.0, ('q0','q2','q1'):-2.0}
```

#### 1-3-3. データ形式変換
| from | to | サンプルコード |
|------|----|----------------|
| PyQUBO | Amplify | `QuData.input().from_pyqubo(expr).to_amplify()` |
| NumPy 配列 | dimod-BQM | `QuData.input().from_array(arr).to_dimod_bqm()` |
| CSV | PuLP | `QuData.input().from_csv('qudata.csv').to_pulp()` |

### データ形式の変換（入力・変換専用）

デバイスへの様々な入力形式を `QuData.input()`（実体は `QdAnnealingIR`）経由で変換できます。**実行（Executor）は不要**です。

#### pyqubo から Amplify への変換
```python
from pyqubo import Binary
from qudas import QuData

# Pyqubo で問題を定義
q0, q1 = Binary("q0"), Binary("q1")
prob = (q0 + q1) ** 2

# QuData に Pyqubo の問題を渡す
qudata = QuData.input().from_pyqubo(prob)
print(qudata.qubo)  # 出力: {('q0', 'q0'): 1.0, ('q0', 'q1'): 2.0, ('q1', 'q1'): 1.0}

# Amplify 形式に変換
amplify_prob = qudata.to_amplify()
print(amplify_prob)
```

#### 配列から BQM への変換
```python
import numpy as np
from qudas import QuData

# Numpy 配列を定義
prob = np.array([
    [1, 1, 0],
    [0, 2, 0],
    [0, 0, -1],
])

# QuData に配列を渡す
qudata = QuData.input().from_array(prob)
print(qudata.qubo)  # 出力: {('q_0', 'q_0'): 1, ('q_0', 'q_1'): 1, ('q_1', 'q_1'): 2, ('q_2', 'q_2'): -1}

# BQM 形式に変換
bqm_prob = qudata.to_dimod_bqm()
print(bqm_prob)
```

#### CSV から PuLP への変換
```python
import pulp
from qudas import QuData

# CSVファイルのパス
csv_file_path = './data/qudata.csv'

# QuData に CSV を渡す
qudata = QuData.input().from_csv(csv_file_path)
print(qudata.qubo)  # 出力: {('q_0', 'q_0'): 1.0, ('q_0', 'q_2'): 2.0, ...}

# PuLP 形式に変換
pulp_prob = qudata.to_pulp()
print(pulp_prob)
```

### データ形式の変換（出力・変換専用）

ソルバ結果などを `QuData.output()`（実体は `QdAnnealingOutput`）へ取り込み、必要なら別形式へ出せます。  
出力側に `.qubo` / `to_amplify()` はありません（入力変換は `QuData.input()` を使います）。

#### PuLP 結果の取り込み
```python
import pulp
from qudas import QuData

prob = pulp.LpProblem("Test_Problem", pulp.LpMinimize)
x = pulp.LpVariable('x', lowBound=0, upBound=1, cat='Binary')
y = pulp.LpVariable('y', lowBound=0, upBound=1, cat='Binary')
prob += 2 * x - y
prob.solve()

out = QuData.output().from_pulp(prob)
print(out.results["block0"]["solution"])  # 変数値
print(out.results["block0"]["energy"])    # 目的関数値
print(out.results["block0"]["device"])    # 'pulp'
```

#### SciPy 結果から dimod SampleSet へ
```python
import numpy as np
from sympy import symbols, lambdify
from scipy.optimize import minimize, Bounds
from qudas import QuData

q0, q1, q2 = symbols('q0 q1 q2')
objective_function = 2 * q0 - q1 - q2
f = lambdify([q0, q1, q2], objective_function, 'numpy')
q = [0.5, 0.5, 0.5]
bounds = Bounds([0, 0, 0], [1, 1, 1])
res = minimize(lambda q: f(q[0], q[1], q[2]), q, method='SLSQP', bounds=bounds)

out = QuData.output().from_scipy(res)
print(out.results["block0"]["solution"])
dimod_samples = out.to_dimod()  # dimod.SampleSet
print(dimod_samples)
```

---

### 1-4. **NEW** 量子ゲート実行 (v0.2 系で追加)
量子ゲート方式の回路を `QdGateExecutor` で実行できます。`qudas[gate]` を入れると Qiskit 系依存が揃い、内部ではシミュレータを呼び出します。

以下ではテストコードと同様に、代表的な 4 パターンの実行／変換例を示します。

#### 1-4-1. 純粋な Qudas での実行
```python
from qudas.gate import (
    QdGateIR, QdGateBlock,
    QdGateInput, QdGateExecutor,
)

# H + CX でベル状態を生成する 2qubit 回路
gates = [
    QdGateIR(gate='h', targets=[0]),
    QdGateIR(gate='cx', targets=[1], controls=[0]),
]
block = QdGateBlock(label='bell', gates=gates, num_qubits=2)
qd_input = QdGateInput(blocks=[block])

# 実行
executor = QdGateExecutor(provider='default')
output = executor.run(qd_input)
print(output.results['bell'])  # => {'counts': {'00': 512, '11': 512}, 'device': 'qiskit'}
```

#### 1-4-2. Qudas の回路 → Qiskit へ変換して実行
```python
from qudas.gate import (
    QdGateIR, QdGateBlock,
    QdGateInput, QdGateExecutor,
)
from qiskit.primitives import Sampler

# H + CX でベル状態を生成する 2qubit 回路
gates = [
    QdGateIR(gate='h', targets=[0]),
    QdGateIR(gate='cx', targets=[1], controls=[0]),
]
block = QdGateBlock(label='bell', gates=gates, num_qubits=2)

ir = block.to_ir()               # QdGateBlock → QdAlgorithmIR
qc = ir.to_qiskit()              # → qiskit.QuantumCircuit
qc.measure_all()

sampler = Sampler()
result = sampler.run([qc], shots=256).result()
counts = result.quasi_dists[0]
print(counts)
```

#### 1-4-3. 外部フレームワーク (Qiskit) の回路 → Qudas で実行
```python
from qiskit import QuantumCircuit
qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)

# Qiskit → QdAlgorithmIR
from qudas.gate.ir import QdAlgorithmIR
ir = QdAlgorithmIR.from_qasm(qc)

# QdAlgorithmIR → QdGateBlock → Qudas 実行
block = QdGateBlock(label='block0', gates=ir.gates, num_qubits=1)
qd_input = QdGateInput(blocks=[block])
output = QdGateExecutor().run(qd_input)
print(output.results["block0"])
```

#### 1-4-4. 外部フレームワーク → さらに別フレームワークへ変換して実行
```python
from qiskit import QuantumCircuit, qasm2
from qiskit.primitives import Sampler

# オリジナル回路
qc_original = QuantumCircuit(1, 1)
qc_original.x(0)
qc_original.measure(0, 0)

# Qiskit → OpenQASM 文字列
qasm_str = qasm2.dumps(qc_original)

# OpenQASM → QdAlgorithmIR (qudas) → Qiskit 再生成
from qudas.gate.ir import QdAlgorithmIR
ir = QdAlgorithmIR.from_qasm(qasm_str)
qc_converted = ir.to_qiskit()

# 別 backend (再度 Qiskit シミュレータ) で実行
sampler = Sampler()
result = sampler.run([qc_converted], shots=128).result()
counts = result.quasi_dists[0]
print(counts)
```

* 複数ブロックを並列に実行したい場合は `QdGateExecutor.run_split()` を利用してください。

### 1-5. **NEW** アニーリング実行 API (v0.2 系で追加)
v0.1 にはアニーリング実行 API は存在しません。v0.2 で新規に追加されました。以下の使用例をご参照ください。

#### 使用例
```python
from qudas.annealing import QdAnnealingInput, QdAnnealingBlock, QdAnnealingExecutor

# QUBO と Block を用意
qubo = {('q0', 'q1'): 1.0, ('q1', 'q1'): -1.0}
block = QdAnnealingBlock(qubo, label='block0')

# 入力 & 実行
qd_input = QdAnnealingInput([block])
executor = QdAnnealingExecutor(provider='default')
output = executor.run(qd_input)

# 結果
solution = output.results["block0"]["solution"]
energy = output.results["block0"]["energy"]
device = output.results["block0"]["device"]
print(solution, energy, device)
```

### 1-6. パイプライン（`QdPipeline` / `Pipeline`）

古典・量子ステップを名前付きでつなぐパイプラインは `qudas.pipeline` の `QdPipeline`（互換別名 `Pipeline`）を利用します。

**サポートする主な操作:** `fit` / `optimize` / `predict` / `get_results` / `get_models` / `set_context`  
**未サポート:** 公開 `transform` / `fit_transform`、および `set_params(step__param=...)` 形式のネスト解決

具体例は [examples/fmqa/main.py](examples/fmqa/main.py)、[examples/vqe/run_vqe.py](examples/vqe/run_vqe.py)、[examples/qsvm-gate/main.py](examples/qsvm-gate/main.py) および [tests/test_pipeline.py](tests/test_pipeline.py) を参照してください。実行結果への統計情報（`qudas.core.statistics`）は v0.2.1 で追加され、本版でも利用できます。

---

## ライセンス
このプロジェクトはApache-2.0ライセンスの下で提供されています。詳細は`LICENSE`ファイルを参照してください。

## 謝辞
本成果は、国立研究開発法人新エネルギー・産業技術総合開発機構 (ＮＥＤＯ) の助成事業として得られたものです。

---

## 開発者向けドキュメント

開発に関する詳細な手順やガイドラインは `DEVELOPER.md` を参照してください。

## テストコード

自動テストは `tests/` にあります。`qudata`・ゲート・アニーリング・パイプラインごとにモジュールを分けています。手動で動作確認するスクリプトは `examples/manual/` を参照してください。

実行方法は `DEVELOPER.md` の「テスト」節を参照してください。
