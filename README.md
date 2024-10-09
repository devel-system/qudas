
# Qudas (Quantum Data Transformation Library)

Qudasは、量子計算における最適化問題の入出力データを変換するためのPythonライブラリです。異なるデータ形式間の変換をサポートし、さまざまな量子計算環境での統一的なデータ処理を可能にします。

## 主な機能
- 量子計算における入力データのフォーマット変換
- 計算結果の出力データのフォーマット変換
- AnnealingやGateデバイスのデータに対応

## インストール
以下のコマンドを使用してインストールします。

```
pip install qudas
```

## 使い方
基本的な使い方は以下の通りです。

```python
import qudas

# 入力データの変換
input_data = {...}
converted_data = qudas.transform(input_data)

# 出力データの変換
output_data = {...}
converted_result = qudas.convert_output(output_data)
```

## 開発者向け情報

### ドキュメントの生成方法

Sphinxを使用してHTMLドキュメントを生成します。

1. 初回の設定
```
mkdir docs
cd docs
sphinx-quickstart
sphinx-apidoc -f -o source ../qudas
```

2. `docs/source/conf.py` を適宜修正

3. ドキュメントをビルド
```
cd docs
make clean
make html
```

生成されたHTMLドキュメントは `docs/build/html/index.html` で確認できます。
Markdownは [GitHub-flavored Markdown](https://guides.github.com/features/mastering-markdown/) を参考にしてください。

### テスト
Qudasのテストは、`tests/` ディレクトリに配置された `test_xxx.py` ファイルで行います。テストの実行は以下のコマンドで可能です。

```
pytest tests/
```

### パッケージの更新方法
以下のコマンドでパッケージを更新します。

```
pip install . -U
```

## ライセンス
このプロジェクトはMITライセンスの下で提供されています。詳細は`LICENSE`ファイルを参照してください。
