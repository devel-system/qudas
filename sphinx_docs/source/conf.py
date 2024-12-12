import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "qudas"))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../..", "examples")
)  # examplesのコードを追加

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Qudas'
copyright = '2024, DEVEL Co., Ltd.'
author = 'KeiichiroHiga'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # ソースコード読み込み用
    'sphinx.ext.viewcode',  # ハイライト済みのソースコードへのリンクを追加
    'sphinx.ext.todo',  # ToDoアイテムのサポート
    'sphinx.ext.napoleon',  # googleスタイルやNumpyスタイルでdocstringを記述した際に必要
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_sidebars = {
    '**': [
        'sidebar/brand.html',
        'sidebar/search.html',
        'sidebar/scroll-start.html',
        'sidebar/navigation.html',
        'sidebar/ethical-ads.html',
        'sidebar/scroll-end.html',
        'sidebar/variant-selector.html',
    ],
}
html_static_path = ['_static']
html_baseurl = 'https://devel-system.github.io/qudas/'
html_css_files = [
    'css/futo.css',  # 必要であれば追加
]
html_js_files = [
    'scripts/furo.js',  # 必要であれば追加
]

# GitHub Pages用のURL構造を明示
html_context = {
    "display_github": True,
    "github_user": "devel-system",  # GitHubユーザー名
    "github_repo": "qudas",        # リポジトリ名
    "github_version": "develop",      # ブランチ名
    "doc_path": "docs",            # 公開ディレクトリ
}

# コードテーマ
# pygments_style = 'autumn'
# pygments_dark_style = 'autumn'
