Annealing
=============================

量子アニーリング（QUBO）を扱うモジュールです。役割は次の二つに分かれます。

1. **変換（canonical）** — :class:`~qudas.annealing.QdAnnealingIR` が QUBO 正規形と ``from_*`` / ``to_*`` の実装本体です。Public 入口としては :class:`~qudas.qudata.QuData` も使えます（Executor 不要）。
2. **実行** — :class:`~qudas.annealing.QdAnnealingBlock` / :class:`~qudas.annealing.QdAnnealingInput` / :class:`~qudas.annealing.QdAnnealingExecutor` / :class:`~qudas.annealing.QdAnnealingOutput`。

全体配置は :doc:`../index` の「変換と実行の関係」図を参照してください。

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   ir
   input
   output
   executor
   block
