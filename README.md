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