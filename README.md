
# Qudas (Quantum Data Transformation Library)

Qudasは、量子計算における最適化問題の入出力データを変換するためのPythonライブラリです。異なるデータ形式間の変換をサポートし、さまざまな量子計算環境での統一的なデータ処理を可能にします。

## 主な機能
- 量子計算における入力データのフォーマット変換
- 計算結果の出力データのフォーマット変換
- AnnealingやGateデバイスのデータに対応

## インストール
以下のコマンドを使用してインストールします。

```
pip install qudas
```

## 使用方法

### 初期化

```python
from qudata import QuData

# 最適化問題の初期化
prob = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
qudata = QuData.input(prob)
print(qudata.prob)  # 出力: {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}

# Noneで初期化した場合
qudata = QuData.input()
print(qudata.prob)  # 出力: {}
```

### 加算・減算・乗算・べき乗

```python
# 辞書形式の問題を加算
qudata1 = QuData.input({('q0', 'q1'): 1.0})
qudata2 = QuData.input({('q0', 'q0'): 2.0})
result = qudata1 + qudata2
print(result.prob)  # 出力: {('q0', 'q1'): 1.0, ('q0', 'q0'): 2.0}

# 辞書形式の問題をべき乗
qudata = QuData.input({('q0', 'q1'): 1.0})
result = qudata ** 2
print(result.prob)  # 出力: {('q0', 'q1'): 1.0, ('q0', 'q2', 'q1'): -2.0}
```

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
print(qudata.prob)  # 出力: {('q0', 'q0'): 1.0, ('q0', 'q1'): 2.0, ('q1', 'q1'): 1.0}

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
print(qudata.prob)  # 出力: {('q_0', 'q_0'): 1, ('q_0', 'q_1'): 1, ('q_1', 'q_1'): 2, ('q_2', 'q_2'): -1}

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
print(qudata.prob)  # 出力: {('q_0', 'q_0'): 1.0, ('q_0', 'q_2'): 2.0, ...}

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
   print(qudata.prob)  # 出力: {'x': 2.0, 'y': -1.0}

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
   print(qudata.prob)  # 出力: {'q0': 2, 'q1': -1, 'q2': -1}

   # Dimod形式に変換
   dimod_prob = qudata.to_dimod_bqm()
   print(dimod_prob)  # 出力: DimodのBQM形式
```

## テストコード
本ライブラリには、以下のようなテストを含めて動作確認を行っています。

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

## 開発者向け情報

### ドキュメントの生成方法

Sphinxを使用してHTMLドキュメントを生成します。

1. 初回の設定
```
mkdir docs
cd docs
sphinx-quickstart
sphinx-apidoc -f -o source ../qudas
```

2. `docs/source/conf.py` を適宜修正

3. ドキュメントをビルド
```
cd docs
make clean
make html
```

生成されたHTMLドキュメントは `docs/build/html/index.html` で確認できます。
Markdownは [GitHub-flavored Markdown](https://guides.github.com/features/mastering-markdown/) を参考にしてください。

### テスト
Qudasのテストは、`tests/` ディレクトリに配置された `test_xxx.py` ファイルで行います。テストの実行は以下のコマンドで可能です。

```
pytest tests/
```

### コードフォーマット

このプロジェクトでは、Python コードのフォーマットに [Black](https://black.readthedocs.io/en/stable/) を使用しています。`Black` は自動でコードを整形し、一貫したスタイルを保つことができます。

#### `Black` を使ったコードフォーマット

1. `Black` のインストール:
    - 開発環境用の依存パッケージとともに `Black` をインストールします:
    ```bash
    pip install .[dev]
    ```

2. コードを自動フォーマットするには、プロジェクトのルートディレクトリで次のコマンドを実行します:
    ```bash
    black .
    ```

    これにより、すべての Python ファイルが自動的にフォーマットされます。

#### コードフォーマットの確認

`black --check .` コマンドを使用すると、コードがフォーマットされているかどうかを確認することができます。このコマンドは実際にはファイルを変更せず、フォーマットが必要かどうかを表示するだけです。

```bash
black --check .
```

### パッケージの更新方法
以下のコマンドでパッケージを更新します。

```
pip install .[dev] -U
```

## ライセンス
このプロジェクトはApache-2.0ライセンスの下で提供されています。詳細は`LICENSE`ファイルを参照してください。
