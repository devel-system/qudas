# Qudas (Quantum datas)

量子計算における入出力データを変換するライブラリ。

You can use
[GitHub-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
to write your content.

## Sphinxの使い方

Viewを作成する。
```
mkdir docs
cd docs
sphinx-quickstart
sphinx-apidoc -f -o docs/source qudas
```

`docs/source/conf.py` を修正

Buildする。

```
cd docs
make html
```

## 更新方法
ドキュメントの更新は `/docs` で `make html` を実行。
パッケージの更新は一番上のディレクトリで `pip install . -U`

## テスト
ファイル名は `test_xxx.py` にする。

## ドキュメント
`qudas/docs/build/html/index.html`