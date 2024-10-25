from setuptools import setup, find_packages

# VERSION ファイルからバージョンを読み込む
with open("VERSION", "r") as version_file:
    version = version_file.read().strip()

# 読み込むREADMEファイルの内容を長い説明として使用
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    "amplify==1.0.1",
    "beautifulsoup4==4.12.3",
    "dimod==0.12.14",
    "dwave-neal==0.6.0",
    "dwave-samplers==1.2.0",
    "joblib==1.3.2",
    "matplotlib==3.7.5",
    "mpmath==1.3.0",
    "networkx==3.1",
    "numpy==1.24.4",
    "pandas==2.0.3",
    "pillow==10.2.0",
    "PuLP==2.8.0",
    "pyqubo==1.4.0",
    "qiskit==1.0.2",
    "qiskit-aer==0.14.1",
    "requests==2.31.0",
    "rustworkx==0.14.2",
    "scikit-learn==1.3.2",
    "scipy==1.10.1",
    "symengine==0.11.0",
    "sympy==1.12",
    "torch==2.2.0",
]

# 開発用ツール
extras_require = {
    "dev": [
        "black==24.8.0",
        "flake8",
        "pytest",
        "python-dotenv==1.0.1",
        "sphinx==7.1.2",
        "sphinx-basic-ng==1.0.0b2",
        "furo==2024.1.29",
    ]
}

setup(
    name="qudas",  # パッケージ名
    version="0.1.0",  # バージョン
    author="Keiichiro Higa",  # 作者
    author_email="higa.devel@gmail.com",  # 作者のメールアドレス
    description="Quantum data transform package",  # パッケージの短い説明
    long_description=long_description,  # 長い説明 (README.mdから取得)
    long_description_content_type="text/markdown",  # 長い説明の形式
    url="https://github.com/devel-system/qudas",  # パッケージのリポジトリURL（適宜変更してください）
    packages=find_packages(include=["qudas", "qudas.*"]),  # パッケージを自動で検出
    classifiers=[  # パッケージに関する分類
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Pythonの必要バージョン
    install_requires=install_requires,  # 本番環境用の依存関係
    extras_require=extras_require,  # 開発環境用の依存関係
    include_package_data=True,  # パッケージデータの自動インクルード
)
