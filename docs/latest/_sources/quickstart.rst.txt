.. Qudas documentation master file, created by
   sphinx-quickstart on Thu Mar 14 05:33:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Quickstart
=================================
qudasは量子計算と古典計算のパイプラインを作成したり、入出力形式を変換したりすることができるライブラリです。

qudasのインストール
~~~~~~~~~~~~~~~~~

.. code-block::

   pip install qudas

ブラックボックス最適化アルゴリズムであるFMQAのパイプライン作成例
---------------------------------

以下のコードは、FMQAのパイプライン作成例です。

.. literalinclude:: ../../examples/fmqa/main.py
   :language: python
   :linenos:

ここで、``TorchFMQA``, ``AnnealFMQA``, ``PipeIteration`` は別途定義する必要があります。\
3つの処理を Pipeline module を用いて実行することができます。

詳しくは ``examples/fmqa/main.py`` を参照。

データ変換例（QuDataInput）
~~~~~~~~~~~~~~~~~

qudasライブラリでは、デバイスへの様々な入力形式のデータを変換する機能も備えています。以下にいくつかのデータ変換例を示します。

pyqubo から Amplify への変換
---------------------------------

.. code-block:: python

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

配列から BQM への変換
---------------------------------

.. code-block:: python

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

CSV から PuLP への変換
---------------------------------

.. code-block:: python

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

データ変換例（QuDataOutput）
~~~~~~~~~~~~~~~~~

qudasライブラリでは、デバイスからの様々な出力形式のデータを変換する機能も備えています。以下にいくつかのデータ変換例を示します。

PuLP から Amplify への変換
---------------------------------

.. code-block:: python

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

SciPy から Dimod への変換
---------------------------------

.. code-block:: python

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

詳しくは ``test/test_qudata.py`` を参照。