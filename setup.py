from setuptools import setup, find_packages

# VERSION ファイルからバージョンを読み込む
with open("VERSION", "r") as version_file:
    version = version_file.read().strip()

# 読み込むREADMEファイルの内容を長い説明として使用
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    "alabaster==0.7.13",
    "amplify==1.0.1",
    "Babel==2.14.0",
    "backports.tarfile==1.2.0",
    "beautifulsoup4==4.12.3",
    "certifi==2024.2.2",
    "charset-normalizer==3.3.2",
    "click==8.1.7",
    "contourpy==1.1.1",
    "cycler==0.12.1",
    "Deprecated==1.2.14",
    "dill==0.3.8",
    "dimod==0.12.14",
    "docutils==0.20.1",
    "dwave-neal==0.6.0",
    "dwave-samplers==1.2.0",
    "filelock==3.13.1",
    "fonttools==4.49.0",
    "fsspec==2024.2.0",
    "idna==3.6",
    "imagesize==1.4.1",
    "importlib-metadata==7.0.1",
    "importlib_resources==6.3.0",
    "jaraco.classes==3.4.0",
    "jaraco.context==6.0.1",
    "jaraco.functools==4.1.0",
    "Jinja2==3.1.3",
    "joblib==1.3.2",
    "keyring==25.4.1",
    "kiwisolver==1.4.5",
    "markdown-it-py==3.0.0",
    "MarkupSafe==2.1.5",
    "matplotlib==3.7.5",
    "mdurl==0.1.2",
    "more-itertools==10.5.0",
    "mpmath==1.3.0",
    "mypy-extensions==1.0.0",
    "networkx==3.1",
    "nh3==0.2.18",
    "numpy==1.24.4",
    "packaging==23.2",
    "pandas==2.0.3",
    "pathspec==0.12.1",
    "pbr==6.0.0",
    "pillow==10.2.0",
    "pkginfo==1.10.0",
    "platformdirs==4.3.6",
    "psutil==5.9.8",
    "PuLP==2.8.0",
    "Pygments==2.17.2",
    "pylatexenc==2.10",
    "pyparsing==3.1.2",
    "pyproject_hooks==1.2.0",
    "pyqubo==1.4.0",
    "python-dateutil==2.8.2",
    "pytz==2024.1",
    "qiskit==1.0.2",
    "qiskit-aer==0.14.1",
    "requests==2.31.0",
    "requests-toolbelt==1.0.0",
    "rfc3986==2.0.0",
    "rich==13.9.3",
    "rustworkx==0.14.2",
    "scikit-learn==1.3.2",
    "scipy==1.10.1",
    "six==1.16.0",
    "snowballstemmer==2.2.0",
    "soupsieve==2.5",
    "symengine==0.11.0",
    "sympy==1.12",
    "threadpoolctl==3.3.0",
    "tomli==2.0.2",
    "torch==2.2.0",
    "tqdm==4.66.2",
    "typing_extensions==4.9.0",
    "tzdata==2023.4",
    "urllib3==2.2.0",
    "wrapt==1.16.0",
    "zipp==3.17.0",
]

# 開発用ツール
extras_require = {
    "dev": [
        "black==24.8.0",
        "build==1.2.2.post1",
        "flake8",
        "pytest",
        "python-dotenv==1.0.1",
        "sphinx==7.1.2",
        "sphinx-basic-ng==1.0.0b2",
        "furo==2024.1.29",
        "twine==5.1.1",
        "readme_renderer==43.0",
    ]
}

setup(
    name="qudas",  # パッケージ名
    version=version,  # バージョン
    author="Keiichiro Higa",  # 作者
    author_email="higa.devel@gmail.com",  # 作者のメールアドレス
    description="Quantum data transform package",  # パッケージの短い説明
    long_description=long_description,  # 長い説明 (README.mdから取得)
    long_description_content_type="text/markdown",  # 長い説明の形式
    url="https://github.com/devel-system/qudas",  # パッケージのリポジトリURL（適宜変更してください）
    packages=find_packages(include=["qudas", "qudas.*"]),  # パッケージを自動で検出
    classifiers=[  # パッケージに関する分類
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Pythonの必要バージョン
    install_requires=install_requires,  # 本番環境用の依存関係
    extras_require=extras_require,  # 開発環境用の依存関係
    include_package_data=True,  # パッケージデータの自動インクルード
)
