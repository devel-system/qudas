# Qudas Developer Guide

このドキュメントでは Qudas の開発手順、テスト、ドキュメント生成、コードスタイルなど、コントリビューションに必要な情報をまとめています。

---

## 1. ドキュメント生成 (Sphinx)

```bash
cd sphinx_docs
make clean && make html
# → build/html/index.html をブラウザで開く
```

---

## 2. テスト

```bash
pytest tests/
```

---

## 3. フォーマット / Lint

```bash
black .
flake8 .
```

- Black: 自動整形
- Flake8: 静的解析 (PEP8 + α)

コミット前に以下を実行してコードスタイルを統一してください。

```bash
pre-commit run --all-files
```

---

## 4. パッケージ更新

```bash
pip install .[dev] -U
```

---

## 5. コントリビュートフロー

1. `develop` ブランチを最新に pull し、新しい topic ブランチを切る。
2. 実装・テストを行い `pytest` が通ることを確認。
3. `pre-commit run --all-files` を実行しフォーマットを適用。
4. Pull Request を作成し、レビュワーへアサイン。
5. CI が Green になったらマージ。

---

## ライセンス
このプロジェクトはApache-2.0ライセンスの下で提供されています。詳細は`LICENSE`ファイルを参照してください。

## 謝辞
本成果は、国立研究開発法人新エネルギー・産業技術総合開発機構 (ＮＥＤＯ) の助成事業として得られたものです。