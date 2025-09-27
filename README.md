
# Qudas (Quantum Data Transformation Library)

Qudasは、量子計算における最適化問題の入出力データを変換するためのPythonライブラリです。異なるデータ形式間の変換をサポートし、さまざまな量子計算環境での統一的なデータ処理を可能にします。

本 README は

1. ライブラリ利用者向けドキュメント (Install / How-to / API 遷移ガイド)
2. 開発者向けドキュメント (開発フロー / コントリビュート手順)

の 2 つのセクションで構成されています。

---

## 1️⃣ ライブラリ利用者向けドキュメント

### 1-1. インストール
```bash
pip install qudas  # PyPI 版 (推奨)
# or
pip install git+https://github.com/devel-system/qudas.git@v0.2.0  # 開発版
```

### 1-2. クイックスタート
以下では代表的なユースケースを抜粋します。詳細は [examples/](examples/) も参照してください。

#### 1-2-1. QuData の生成
```python
from qudas import QuData

# QUBO (dict) から生成
qubo = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
qudata = QuData.input(qubo)
print(qudata.prob)               # => {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
```

#### 1-2-2. 四則演算
```python
q1 = QuData.input({('q0', 'q1'): 1.0})
q2 = QuData.input({('q0', 'q0'): 2.0})
print((q1 + q2).prob)            # => {('q0','q1'):1.0, ('q0','q0'):2.0}
print((q1 ** 2).prob)            # => {('q0','q1'):1.0, ('q0','q2','q1'):-2.0}
```

#### 1-2-3. データ形式変換
| from | to | サンプルコード |
|------|----|----------------|
| PyQUBO | Amplify | `QuData.input().from_pyqubo(expr).to_amplify()` |
| NumPy 配列 | dimod-BQM | `QuData.input().from_array(arr).to_dimod_bqm()` |
| CSV | PuLP | `QuData.input().from_csv('qudata.csv').to_pulp()` |

### データ形式の変換（QuDataInput）
デバイスへの様々な入力形式のデータを `QuData` オブジェクトを介して変換することができます。

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

### データ形式の変換（QuDataOutput）
デバイスからの様々な出力形式のデータを `QuData` オブジェクトを介して変換することができます。

#### PuLP から Amplify への変換
```python
import pulp
from qudas import QuData

# PuLP問題を定義して解く
prob = pulp.LpProblem("Test Problem", pulp.LpMinimize)
x = pulp.LpVariable('x', lowBound=0, upBound=1, cat='Binary')
y = pulp.LpVariable('y', lowBound=0, upBound=1, cat='Binary')
prob += 2*x - y
prob.solve()

# QuDataOutputのインスタンスを生成し、from_pulpメソッドで問題を変換
qudata = QuData.output().from_pulp(prob)
print(qudata.qubo)  # 出力: {'x': 2.0, 'y': -1.0}

# Amplify形式に変換
amplify_prob = qudata.to_amplify()
print(amplify_prob)  # 出力: Amplifyの目標関数形式
```

#### SciPy から Dimod への変換
```python
import numpy as np
from sympy import symbols, lambdify
from scipy.optimize import minimize, Bounds
from qudas import QuData

# シンボリック変数の定義
q0, q1, q2 = symbols('q0 q1 q2')

# 目的関数を定義
objective_function = 2 * q0 - q1 - q2

# シンボリック関数を数値化して評価できる形式に変換
f = lambdify([q0, q1, q2], objective_function, 'numpy')

# 初期解 (すべて0.5に設定)
q = [0.5, 0.5, 0.5]

# バイナリ変数の範囲を定義 (0 <= x <= 1)
bounds = Bounds([0, 0, 0], [1, 1, 1])

# SciPyで制約付き最適化を実行
res = minimize(lambda q: f(q[0], q[1], q[2]), q, method='SLSQP', bounds=bounds)

# QuDataOutputのインスタンスを生成し、from_scipyメソッドをテスト
qudata = QuData.output().from_scipy(res)
print(qudata.qubo)  # 出力: {'q0': 2, 'q1': -1, 'q2': -1}

# Dimod形式に変換
dimod_prob = qudata.to_dimod_bqm()
print(dimod_prob)  # 出力: DimodのBQM形式
```

---

### 1-3. **NEW** 量子ゲート実行 (v0.2 系で追加)
量子ゲート方式の回路を `QdGateExecutor` で実行できます。内部で Qiskit シミュレータを呼び出すため、追加依存は `qiskit` のみです。

以下ではテストコードと同様に、代表的な 4 パターンの実行／変換例を示します。

#### 1-3-1. 純粋な Qudas での実行
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
block = QdGateBlock(name='bell', gates=gates, num_qubits=2)
qd_input = QdGateInput(blocks=[block])

# 実行
executor = QdGateExecutor(provider='default')
output = executor.run(qd_input)
print(output.results['bell'])  # => {'counts': {'00': 512, '11': 512}, 'device': 'qiskit'}
```

#### 1-3-2. Qudas の回路 → Qiskit へ変換して実行
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

#### 1-3-3. 外部フレームワーク (Qiskit) の回路 → Qudas で実行
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

#### 1-3-4. 外部フレームワーク → さらに別フレームワークへ変換して実行
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

---

### 1-4. **NEW** アニーリング実行 API (v0.2 系で追加)
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

## ライセンス
このプロジェクトはApache-2.0ライセンスの下で提供されています。詳細は`LICENSE`ファイルを参照してください。

## 謝辞
本成果は、国立研究開発法人新エネルギー・産業技術総合開発機構 (ＮＥＤＯ) の助成事業として得られたものです。

---

## 開発者向けドキュメント

開発に関する詳細な手順やガイドラインは `DEVELOPER.md` を参照してください。

## テストコード
```python
class TestQudata(unittest.TestCase):

    def test_init_with_dict(self):
        # 辞書データで初期化する場合のテスト
        prob = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
        qudata = QuData.input(prob)
        self.assertTrue(dicts_are_equal(qudata.prob, prob))

    def test_add(self):
        # __add__メソッドのテスト
        prob1 = {('q0', 'q1'): 1.0}
        prob2 = {('q0', 'q0'): 2.0}
        qudata1 = QuData.input(prob1)
        qudata2 = QuData.input(prob2)
        result = qudata1 + qudata2
        expected = {('q0', 'q1'): 1.0, ('q0', 'q0'): 2.0}
        self.assertTrue(dicts_are_equal(result.prob, expected))
```
