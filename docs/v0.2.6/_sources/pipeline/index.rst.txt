パイプライン
=================================

量子・古典をまたぐ処理の骨格は :class:`~qudas.pipeline.QdPipeline`（エイリアス名 ``Pipeline``）です。

**サポートする主な操作**

* ``fit`` / ``optimize`` / ``predict``
* ``get_results`` / ``get_models``
* ``set_context`` / ``get_context`` （および互換の global_params 系）

**未サポート**

* 公開メソッドとしての ``transform`` / ``fit_transform``
* ``set_params`` の step ネスト指定（例: step 名とパラメータ名の結合形式）

ステップ実装のベースや反復・最適化用のミキサンは下記 toctree を参照してください。

.. autoclass:: qudas.pipeline.QdPipeline
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   steps/base
   steps/iterator_mixin
   steps/optimizer_mixin
