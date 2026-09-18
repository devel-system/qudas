# Qudas Developer Guide

このドキュメントでは Qudas の開発手順、テスト、ドキュメント生成、コードスタイルなど、コントリビューションに必要な情報をまとめています。

---

## 1. ドキュメント生成 (Sphinx)

```bash
pip install -e ".[dev]"
cd sphinx_docs
make clean && make html
# → build/html/index.html をブラウザで開く
```

公開用に `docs/latest/` へ反映する場合の例:

```bash
rm -rf docs/latest
cp -R sphinx_docs/build/html docs/latest
# 必要なら固定スナップショットも作成:
# rm -rf docs/v0.2.6 && cp -R sphinx_docs/build/html docs/v0.2.6
# docs/index.html の版リンクを更新
```

---

## 2. テスト

開発用依存とオプション機能一式を入れたうえで実行してください。

```bash
pip install -e ".[dev,all]"
pytest
```

### レイヤ

| レイヤ | 場所 | 備考 |
|--------|------|------|
| 正の契約（推奨） | `tests/qudata/test_qudata_*.py`、`tests/test_*.py` | facade / Executor / Pipeline |
| レガシー互換 | `tests/qudata/legacy/` | Deprecated `QuDataInput`/`Output`。`pytest -m legacy` |
| 手動スモーク | `examples/manual/` | pytest 対象外 |

レガシーを除いて正経路だけ回す例:

```bash
pytest -m "not legacy"
```

---

## 3. フォーマット / Lint

```bash
black .
flake8 .
```

- Black: `pyproject.toml` の `[tool.black]`（CI では `black --check .`）
- Flake8: `dev` extra に含まれます。リポジトリに専用設定ファイルは現時点でありません

CI（`.github/workflows/python-ci.yml`）では **pytest** と **black --check** を実行します。  
`pre-commit` 設定（`.pre-commit-config.yaml`）は現時点でリポジトリに含まれていません。導入する場合は別途追加してください。

---

## 4. パッケージ更新

```bash
pip install -e ".[dev]" -U
```

---

## 5. コントリビュートフロー

1. `develop` ブランチを最新に pull し、新しい topic ブランチを切る。
2. 実装・テストを行い `pytest` が通ることを確認。
3. `black .`（必要なら `flake8 .`）でスタイルを確認。
4. Pull Request を作成し、レビュワーへアサイン。
5. CI が Green になったらマージ。

---

## ライセンス
このプロジェクトはApache-2.0ライセンスの下で提供されています。詳細は`LICENSE`ファイルを参照してください。

## 謝辞
本成果は、国立研究開発法人新エネルギー・産業技術総合開発機構 (ＮＥＤＯ) の助成事業として得られたものです。
