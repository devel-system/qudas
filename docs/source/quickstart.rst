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

.. code-block:: python

    # qudas & pipline
    from qudas.pipeline import Pipeline
    from torch_fmqa import TorchFMQA
    from anneal_fmqa import AnnealFMQA
    from pipe_iteration import PipeIteration
    from utils import make_blackbox_func, init_training_data

    # module
    import torch

    # 適当な関数を作成 (d次元, y = xQx)
    d = 100
    blackbox = make_blackbox_func(d)

    # 初期教師データの数
    N0 = 60
    x, y = init_training_data(d, N0, blackbox)
    print(f"{x.shape=}, {y.shape=}")

    # FMQA サイクルの実行回数
    N = 10

    # 初期パラメータ
    k = 10
    v = torch.randn((d, k), requires_grad=True)
    w = torch.randn((d,), requires_grad=True)
    w0 = torch.randn((), requires_grad=True)
    parameters = {'v': v, 'w': w, 'w0': w0}

    # pipeline
    steps = [
        ('TorchFMQA', TorchFMQA()),
        ('AnnealFMQA', AnnealFMQA(blackbox, d, token="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")),
        ('pipeIteration', PipeIteration(blackbox, d, loop_num=N))
    ]

    pipe = Pipeline(steps)
    pipe.set_global_params(parameters)

    # 最適化
    result = pipe.optimize(x, y)
    print(f"{result=}")

ここで、``TorchFMQA``, ``AnnealFMQA``, ``PipeIteration`` は別途定義する必要があります。\
3つの処理を Pipeline module を用いて実行することができます。

詳しくは ``examples/test_fmqa.py`` を参照。

データ変換例
~~~~~~~~~~~~~~~~~

qudasライブラリでは、様々な形式のデータを変換する機能も備えています。以下にいくつかのデータ変換例を示します。

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

詳しくは ``test/test_qudata.py`` を参照。