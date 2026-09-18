Quickstart
=================================

Qudas は、:doc:`index` で示したとおり **パイプライン**・**形式変換（QuData）**・**ゲート／アニーリング実行（任意）** を組み合わせて使います。ここではインストールと代表的なコード例だけをまとめます。

インストール
~~~~~~~~~~~~

コアのみ、または用途に応じた extra を指定します。

.. code-block:: bash

   pip install qudas                 # コア（numpy + matplotlib）
   pip install "qudas[annealing]"    # 変換 + アニーリング実行一式
   pip install "qudas[gate]"         # Qiskit ベースのゲート実行
   pip install "qudas[all]"          # gate + annealing

変換だけ使いたい場合も、現状は変換用 SDK が ``annealing`` extra に含まれます（実行用依存との分離は今後検討）。

ブラックボックス最適化（FMQA）のパイプライン作成例
----------------------------------------------------

以下のコードは、FMQAのパイプライン作成例です。

.. literalinclude:: ../../examples/fmqa/main.py
   :language: python
   :linenos:

ここで、``TorchFMQA``, ``AnnealFMQA``, ``PipeIteration`` は別途定義する必要があります。\
3つの処理を Pipeline module を用いて実行することができます。

詳しくは ``examples/fmqa/main.py`` を参照。

データ変換例（入力・変換専用）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``QuData.input()`` は :class:`~qudas.annealing.QdAnnealingIR` を返します。Executor は不要です。

pyqubo から Amplify への変換
---------------------------------

.. code-block:: python

   from pyqubo import Binary
   from qudas import QuData

   # Pyqubo で問題を定義
   q0, q1 = Binary("q0"), Binary("q1")
   prob = (q0 + q1) ** 2

   # QuData に Pyqubo の問題を渡す（実体は QdAnnealingIR）
   qudata = QuData.input().from_pyqubo(prob)
   print(qudata.qubo)  # または qudata.prob

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
   print(qudata.qubo)

   # BQM 形式に変換
   bqm_prob = qudata.to_dimod_bqm()
   print(bqm_prob)

CSV から PuLP への変換
---------------------------------

.. code-block:: python

   from qudas import QuData

   # CSVファイルのパス
   csv_file_path = './data/qudata.csv'

   # QuData に CSV を渡す
   qudata = QuData.input().from_csv(csv_file_path)
   print(qudata.qubo)

   # PuLP 形式に変換
   pulp_prob = qudata.to_pulp()
   print(pulp_prob)

データ変換例（出力・変換専用）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``QuData.output()`` は :class:`~qudas.annealing.QdAnnealingOutput` を返します。  
出力側に ``.qubo`` / ``to_amplify()`` はありません（入力変換は ``QuData.input()`` を使います）。

PuLP 結果の取り込み
---------------------------------

.. code-block:: python

   import pulp
   from qudas import QuData

   prob = pulp.LpProblem("Test_Problem", pulp.LpMinimize)
   x = pulp.LpVariable('x', lowBound=0, upBound=1, cat='Binary')
   y = pulp.LpVariable('y', lowBound=0, upBound=1, cat='Binary')
   prob += 2 * x - y
   prob.solve()

   out = QuData.output().from_pulp(prob)
   print(out.results["block0"]["solution"])
   print(out.results["block0"]["energy"])
   print(out.results["block0"]["device"])

SciPy 結果から dimod SampleSet へ
---------------------------------

.. code-block:: python

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
   dimod_samples = out.to_dimod()
   print(dimod_samples)

詳しくは ``tests/qudata/test_qudata_facade.py`` および :doc:`annealing/ir` / :doc:`annealing/output` を参照。

アニーリング実行例
~~~~~~~~~~~~~~~~~~

変換とは別経路です。``QdAnnealingExecutor`` を使います。

.. code-block:: python

   from qudas.annealing import QdAnnealingInput, QdAnnealingBlock, QdAnnealingExecutor

   qubo = {('q0', 'q1'): 1.0, ('q1', 'q1'): -1.0}
   block = QdAnnealingBlock(qubo, label='block0')
   qd_input = QdAnnealingInput([block])
   output = QdAnnealingExecutor(provider='default').run(qd_input)
   print(output.results["block0"]["solution"], output.results["block0"]["energy"])
