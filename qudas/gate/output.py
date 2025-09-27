from qudas.core.output_base import QdOutputBase, QdOutputBaseData
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class QdGateOutputData(QdOutputBaseData):
    counts: Dict[str, int]
    expectation_value: Optional[float] = None
    shots: Optional[int] = None


class QdGateOutput(QdOutputBase):
    """量子ゲート方式の計算結果を保持するアウトプットクラス。

    1 ブロックにつき 1 つの結果辞書を保持し、複数ブロック分を
    `results` という大域辞書で管理する設計とする。

    Example
    -------
    >>> results = {
    ...     "blockA": {
    ...         "solution": {"x0": 1, "x1": 0},
    ...         "counts": {"00": 100, "11": 200},
    ...         "expectation_value": 0.5,
    ...         "shots": 300,
    ...         "device": "aer_simulator",
    ...     },
    ...     "blockB": {
    ...         "solution": {"x0": 0, "x1": 1},
    ...         "counts": {"00": 150, "11": 150},
    ...         "expectation_value": 0.2,
    ...         "shots": 300,
    ...         "device": "braket_ionq",
    ...     },
    ... }
    >>> qd_out = QdGateOutput(results)
    >>> qd_out.get_block_solution("blockA")
    {'x0': 1, 'x1': 0}
    """

    def __init__(self, results: Optional[Dict[str, QdGateOutputData]] = None):
        self.results = results or {}

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

    @classmethod
    def from_sdk_format(cls, sdk_obj: Any, target: str) -> "QdGateOutput":
        return cls(sdk_obj["results"])

    def visualize(self):  # noqa: D401 – simple method name
        """結果を簡易可視化 (テキスト出力)。"""

        try:
            import matplotlib.pyplot as plt  # type: ignore

            for idx, (label, res) in enumerate(
                self.results.items()
                if isinstance(self.results, dict)
                else [("", self.results)]
            ):
                plt.figure(idx)
                if "counts" in res:
                    plt.bar(res["counts"].keys(), res["counts"].values())
                    plt.title(f"Counts for {label}")
            plt.show()
        except Exception:
            # matplotlib 無い場合、テキスト表示にフォールバック
            print(
                "QuDataGateOutput.visualize(): matplotlib が見つからないためテキスト出力します。"
            )
            print(self.results)


QdGateOut = QdGateOutput
