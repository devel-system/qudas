QuData クラス
=================================

``QuData.input()`` は :class:`~qudas.annealing.QdAnnealingIR` を、``QuData.output()`` は :class:`~qudas.annealing.QdAnnealingOutput` を返します。
変換メソッド（``from_*`` / ``to_*``）の詳細は annealing 側の API を参照してください。

.. warning::

   ``QuData.input(mode="gate")`` はサポートしていません。回路は :doc:`../gate/index` を使用してください。

.. autoclass:: qudas.qudata.QuData
   :members:
   :undoc-members:
   :show-inheritance:
