from qudas.core.base import QdOutBase


class QuDataGateOutput(QdOutBase):
    def __init__(self, results):
        self.results = results

    def to_dict(self):
        return self.results

    # --------------------------------------------------------------
    # 抽象メソッド実装
    # --------------------------------------------------------------
    def to_sdk_format(self, target: str):  # noqa: D401 – simple method name
        """Backend 依存フォーマットへ変換 (ダミー実装)。"""

        # 本実装では target に応じたフォーマット変換を行う。
        # 未サポートの場合でも呼び出しエラーとならないよう
        # とりあえず内部辞書をそのまま返す。
        return {"target": target, "results": self.results}

    def visualize(self):  # noqa: D401 – simple method name
        """結果を簡易可視化 (テキスト出力)。"""

        try:
            import matplotlib.pyplot as plt  # type: ignore

            for idx, (label, res) in enumerate(self.results.items() if isinstance(self.results, dict) else [("", self.results)]):
                plt.figure(idx)
                if "counts" in res:
                    plt.bar(res["counts"].keys(), res["counts"].values())
                    plt.title(f"Counts for {label}")
            plt.show()
        except Exception:
            # matplotlib 無い場合、テキスト表示にフォールバック
            print("QuDataGateOutput.visualize(): matplotlib が見つからないためテキスト出力します。")
            print(self.results)


QdGateOut = QuDataGateOutput