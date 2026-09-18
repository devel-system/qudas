.. _whatsnew-0-2-6:

Qudas 0.2.6 の変更点
================================

リリース日: 2026-09-18（詳細はリポジトリ内 ``CHANGELOG.md`` の 0.2.6 節を参照）

追加
----

- ``QuData`` facade 向け契約テスト
- Sphinx リリースノートの整備とドキュメント再ビルド

変更
----

- **形式変換:** ``QuData`` を変換専用 Public 入口、実装の正を ``QdAnnealingIR`` / ``QdAnnealingOutput`` として明示。実行は Executor 別経路
- ``QuData.input(mode="gate")`` は未サポート（``NotImplementedError``）。回路は ``qudas.gate`` を使用
- Pipeline のサポート / 未サポート API を現行実装に合わせて記載
- packaging（``find_packages``）と ``__version__`` 解決の改善

非推奨
------

- ``QuDataInput`` / ``QuDataOutput`` （v0.3 削除予定）。移行先は ``QuData`` または annealing 側クラス

修正
----

- ``qudas.core`` の新名 export
- Executor naive fallback 時の警告
- CI / 開発者向け手順の実態合わせ

関連: :doc:`whatsnew_0_2_5`
