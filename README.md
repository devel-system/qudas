
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

### データ形式の変換
さまざまな形式のデータを `QuData` オブジェクトを介して変換することができます。

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

### パッケージの更新方法
以下のコマンドでパッケージを更新します。

```
pip install . -U
```

## ライセンス
このプロジェクトはMITライセンスの下で提供されています。詳細は`LICENSE`ファイルを参照してください。
