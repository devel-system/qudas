.. Qudas documentation master file, created by
   sphinx-quickstart on Thu Mar 14 05:33:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Quickstart
=================================
qudasは量子計算と古典計算のパイプラインを作成したり、入出力形式を変換したりすることができるライブラリです。

qudasのインストール

.. code-block::

   pip install qudas

ブラックボックス最適化アルゴリズムであるFMQAのパイプライン作成例

.. code-block:: python

   # 適当な関数を作成 (d次元, y = xQx)
   d = 100
   blackbox = make_blackbox_func(d)

   # 初期教師データの数
   N0 = 60
   x, y = init_training_data(d, N0)
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
      ('AnnealFMQA', AnnealFMQA(blackbox, d, token="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")),
      ('pipeIteration', PipeIteration(blackbox, d, loop_num=N))
   ]

   pipe = Pipeline(steps)
   pipe.set_grobal_params(parameters)

   # 学習
   result = pipe.optimize(x, y)
   print(f"{result=}")

ここで、``TorchFMQA``, ``AnnealFMQA``, ``PipeIteration`` は別途定義する必要があります。\
3つの処理を Pipeline module を用いて実行することができます。

詳しくは ``test/test_fmqa.py`` を参照。