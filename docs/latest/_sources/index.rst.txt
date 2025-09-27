.. Qudas documentation master file, created by
   sphinx-quickstart on Thu Mar 14 05:33:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Qudas's documentation!
=================================
Qudasは量子計算と古典計算を組み合わせたアルゴリズムを効率よく実行するためのライブラリとなっています。

役割としては大きく2点あります。

- 量子計算と古典計算を行うパイプライン処理
- 量子計算のデータ形式と古典形式のデータ形式の変換

量子計算と古典計算を行うパイプライン処理
------------------------------------------------------------

scikit learnやpytorchには前処理やモデルのフローを定義する pipline の仕組みが存在する。
これに着想を得て、量子・AIのpipelineを開発した。

.. image:: ./img/fmqa_pipeline.png
   :scale: 50%

上図は量子とAIを用いたブラックボックス最適化を行うためのフローである。

フローを記述するためのベースクラスを駆使することで、量子とAIのそれぞれのJobを切り分けることができる。


量子計算のデータ形式と古典形式のデータ形式の変換
------------------------------------------------------------

以下のように各データ形式とそれらを扱うフレームワークを整理した。

.. image:: ./img/data_abstract.png

各モデル毎に対応するオブジェクト形式が異なるので、ライブラリを用いて任意のオブジェクト形式からモデルを変換することができる。\
また、一部のモデルを変換することができる。

.. toctree::
   :maxdepth: 2
   :caption: クイックスタート

   quickstart

.. toctree::
   :maxdepth: 2
   :caption: 入出力変換

   core/index
   gate/index
   annealing/index

.. toctree::
   :maxdepth: 2
   :caption: パイプライン

   pipeline/index

.. toctree::
   :maxdepth: 2
   :caption: その他

   qudata/index

.. toctree::
   :maxdepth: 2
   :caption: 参照

   references

「本ライブラリは、国立研究開発法人新エネルギー・産業技術総合開発機構（ＮＥＤＯ）の委託業務の結果得られたものです。」