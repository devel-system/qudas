QuData
=================================

:class:`~qudas.qudata.QuData` は、最適化・QUBO まわりの **変換専用 Public 入口** です。

* **入口:** ``QuData.input()`` / ``QuData.output()``
* **実装の正:** :class:`~qudas.annealing.QdAnnealingIR` （入力変換・演算）および :class:`~qudas.annealing.QdAnnealingOutput` （結果取り込み）
* **実行は別経路:** :doc:`../annealing/index` の Executor。変換だけなら不要です
* **回路変換は対象外:** :doc:`../gate/index` を参照

.. note::

   レガシーの :class:`~qudas.qudata.QuDataInput` / :class:`~qudas.qudata.QuDataOutput` は **Deprecated** です（v0.3 で削除予定）。
   新規コードでは ``QuData`` または annealing 側クラスを使ってください。

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   main
   base
   input
   output
