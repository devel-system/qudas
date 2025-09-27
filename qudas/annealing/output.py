from qudas.core.output_base import QdOutputBase, QdOutputBaseData
from dataclasses import dataclass
from typing import Dict, Any, Optional

# 依存ライブラリはローカル import で遅延読み込み
# NOTE: 旧 API 互換を保ちつつ多ブロック対応させる。
#   - 旧: `result`/`solution` 単一ブロック辞書を保持し `.result`, `.solution`, `.result_type`
#   - 新: 複数ブロックを `results` 辞書で保持


@dataclass
class QdAnnealingOutputData(QdOutputBaseData):
    energy: float


class QdAnnealingOutput(QdOutputBase):
    """アニーリング系の計算結果を保持するアウトプットクラス。

    1 ブロックにつき 1 つの結果辞書を保持し、複数ブロック分を
    `results` という大域辞書で管理する設計とする。

    Example
    -------
    >>> results = {
    ...     "blockA": {
    ...         "solution": {"x0": 1, "x1": 0},
    ...         "energy": -1.23,
    ...         "device": "amplify",
    ...     },
    ...     "blockB": {
    ...         "solution": {"x0": 0, "x1": 1},
    ...         "energy": -0.98,
    ...         "device": "dimod",
    ...     },
    ... }
    >>> qd_out = QdAnnealingOutput(results)
    >>> qd_out.get_block_solution("blockA")
    {'x0': 1, 'x1': 0}
    """

    def __init__(self, results: Optional[Dict[str, QdAnnealingOutputData]] = None):
        """コンストラクタ。

        Parameters
        ----------
        results : dict[str, dict[str, Any]], optional
            ブロックラベルをキーに、各ブロックの計算結果辞書を
            値として持つ辞書。省略時は空辞書で初期化される。
        """
        self.results = results or {}

    @property
    def solution(self) -> Optional[Any]:
        """最初のブロックの solution を返す（辞書 or None）"""
        if not self.results:
            return None
        return next(iter(self.results.values()))["solution"]

    @property
    def last_device(self) -> Optional[str]:
        if not self.results:
            return None
        return next(reversed(self.results.values()))["device"]

    @property
    def result_type(self) -> Optional[str]:
        return self.last_device

    # ------------------------------------------------------------------
    # 汎用ユーティリティ
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Dict[str, Any]]:  # noqa: D401 – 単純メソッド
        """内部保持している結果辞書をそのまま返す。"""
        return self.results

    def get_block_solution(self, block_label: str):
        """指定したブロックラベルの *solution* を取得する。無ければ None。"""
        return self.results.get(block_label, {}).get('solution', None)

    # ------------------------------------------------------------------
    # 旧 API プロパティ互換
    # ------------------------------------------------------------------
    @property
    def result(self) -> Dict[str, Any]:
        """旧 API 互換: 最初のブロックを {'variables', 'objective'} 形式で返す。"""
        if not self.results:
            return {}
        first_block = self.results[next(iter(self.results))]
        return {
            'variables': first_block.get('solution', {}),
            # energy -> objective 名前変換
            'objective': first_block.get('energy'),
        }

    # ------------------------------------------------------------------
    # 内部ユーティリティ
    # ------------------------------------------------------------------
    def _infer_last_device(self) -> Optional[str]:
        """最新ブロックの device を取得 (存在すれば)"""
        if not self.results:
            return None
        last_block_label = next(reversed(self.results))  # py>=3.8 insertion-order dict
        return self.results[last_block_label].get('device')

    # ------------------------------------------------------------------
    # from_* 系 (外部ライブラリ → QuDataAnnealingOutput)
    # ------------------------------------------------------------------
    def _set_block(
        self, block_label: str, variables: Dict[str, Any], objective: Any, **extras
    ):
        """内部ユーティリティ: 1 ブロック分の結果を書き込む。"""
        self.results[block_label] = {
            'solution': variables,
            'energy': objective,
            **extras,
        }
        return self

    @classmethod
    def from_sdk_format(cls, sdk_obj: Any, target: str) -> "QdAnnealingOutput":
        """外部ライブラリ向けのフォーマットからインスタンスを生成します。

        Args:
            sdk_obj (Any): 外部ライブラリ向けのフォーマット。
            target (str): 外部ライブラリの名前。

        Raises:
            ValueError: サポートされていない外部ライブラリの場合。

        Returns:
            QdAnnealingOutput: インスタンス。
        """
        if target == "pulp":
            return cls.from_pulp(sdk_obj)
        elif target == "amplify":
            return cls.from_amplify(sdk_obj)
        elif target == "dimod":
            return cls.from_dimod(sdk_obj)
        elif target == "scipy":
            return cls.from_scipy(sdk_obj)
        else:
            raise ValueError(f"Unsupported SDK target: {target}")

    def from_pulp(self, problem, block_label: str = 'block0'):
        from pulp import value  # local import

        objective_value = value(problem.objective)
        variables = {var.name: var.value() for var in problem.variables()}
        return self._set_block(block_label, variables, objective_value, device='pulp')

    def from_amplify(self, result, block_label: str = 'block0'):
        variables = {str(k): v for k, v in result.best.values.items()}
        return self._set_block(
            block_label, variables, result.best.objective, device='amplify'
        )

    def from_dimod(self, result, block_label: str = 'block0'):
        return self._set_block(
            block_label, result.first.sample, result.first.energy, device='dimod'
        )

    def from_scipy(self, result, block_label: str = 'block0'):
        import numpy as np  # noqa: F401 – 型検査用に保持

        variables = {f"q{i}": v for i, v in enumerate(result.x)}
        return self._set_block(block_label, variables, result.fun, device='scipy')

    # ------------------------------------------------------------------
    # to_* 系 (QdAnnealingOutput → 外部ライブラリ)
    # ------------------------------------------------------------------
    def to_sdk_format(self, target: str) -> Dict[str, Any]:
        """外部ライブラリ向けのフォーマットに変換します。

        Args:
            target (str): 外部ライブラリの名前。

        Raises:
            ValueError: サポートされていない外部ライブラリの場合。

        Returns:
            dict: 外部ライブラリ向けのフォーマット。
        """
        target = target.lower()
        if target == "dimod":
            return {label: self.to_dimod(label) for label in self.results}
        elif target == "scipy":
            return {label: self.to_scipy(label) for label in self.results}
        else:
            raise ValueError(f"Unsupported SDK target: {target}")

    def to_dimod(self, block_label: str = 'block0'):
        import dimod

        if block_label not in self.results:
            raise KeyError(f"block_label '{block_label}' は存在しません。")
        block = self.results[block_label]
        sampleset = dimod.SampleSet.from_samples(
            samples_like=dimod.as_samples(block["solution"]),
            vartype='BINARY',
            energy=block["energy"],
        )
        return sampleset

    def to_scipy(self, block_label: str = 'block0'):
        from scipy.optimize import OptimizeResult
        import numpy as np

        if block_label not in self.results:
            raise KeyError(f"block_label '{block_label}' は存在しません。")
        block = self.results[block_label]
        x = np.array(list(block["solution"].values()))
        result = OptimizeResult(
            x=x,
            fun=block["energy"],
            success=True,
            status=0,
            message='Optimization terminated successfully.',
            nfev=0,
            nit=0,
        )
        return result

    def visualize(self):
        """結果を可視化します。"""
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


# エイリアス（旧クラス名を残しておく）
QdAnnOut = QdAnnealingOutput
