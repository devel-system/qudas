# Changelog

## [0.1.0] - 2024-10-16
### Added
- パッケージの基本的な構成を作成（`qudas`ディレクトリにメインコードを配置）。
- `examples`ディレクトリにサンプルコードを追加。
- ドキュメントを`docs`ディレクトリに配置。
- ユニットテストを`test`ディレクトリに追加。
- パッケージメタデータの定義を`pyproject.toml`に追加。
- ライセンス情報を`LICENSE`ファイルに記載。
- パッケージの概要を`README.md`に記載。

### Fixed
- 初期セットアップ時のディレクトリ構造とファイル配置に関する修正。

### Changed
- 初期リリース。

---

## [0.2.0] - 2025-07-18
### Added
- 量子ゲート方式に対応する `qudas.gate` パッケージを追加し、`QdGateExecutor` などの主要 API を実装。
- `README.md` を「利用者向け」「開発者向け」に分割し、クイックスタート & API 遷移ガイドを追加。
- `run_split()` によるブロック並列実行機能を Gate Executor に実装。

### Changed
- アニーリング API を刷新: `AnnealingExecutor.execute()` → `QdAnnealingExecutor.run()`、戻り値を dict から `QdAnnealingOutput` へ変更。
- `VERSION` を `0.1.5` → `0.2.0` に更新。

### Deprecated / Removed
- 旧 `AnnealingExecutor` クラスは非推奨。v0.3 で削除予定。

### Fixed
- 小規模なバグ修正および型ヒントの改善。

---

## 今後のバージョン（0.1.0 以降）の予定
### Planned for Future Release
- 新しい機能や修正内容を記載していきます。
