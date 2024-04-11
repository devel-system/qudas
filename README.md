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

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "qudas"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Qudas'
copyright = '2024, DEVEL Co., Ltd.'
author = 'KeiichiroHiga'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # ソースコード読み込み用
    'sphinx.ext.viewcode',  # ハイライト済みのソースコードへのリンクを追加
    'sphinx.ext.todo',  # ToDoアイテムのサポート
    'sphinx.ext.napoleon' #googleスタイルやNumpyスタイルでdocstringを記述した際に必要
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']