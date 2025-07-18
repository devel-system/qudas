from qudas.core.base import QdOutBase
from typing import Dict, Any, Optional

# 依存ライブラリはローカル import で遅延読み込み


# NOTE: 旧 API 互換を保ちつつ多ブロック対応させる。
#   - 旧: `result`/`solution` 単一ブロック辞書を保持し `.result`, `.solution`, `.result_type`
#   - 新: 複数ブロックを `results` 辞書で保持

class QuDataAnnealingOutput(QdOutBase):
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
    >>> qd_out = QuDataAnnealingOutput(results)
    >>> qd_out.get_block_solution("blockA")
    {'x0': 1, 'x1': 0}
    """

    def __init__(
        self,
        results: Optional[Dict[str, Dict[str, Any]]] = None,
        # 旧 API 用
        result: Optional[Dict[str, Any]] = None,
        result_type: Optional[str] = None,
        **kwargs,
    ):
        """コンストラクタ。

        Parameters
        ----------
        results : dict[str, dict[str, Any]], optional
            ブロックラベルをキーに、各ブロックの計算結果辞書を
            値として持つ辞書。省略時は空辞書で初期化される。
        """

        # 旧 API: `result` or `solution` キーワードがあれば `block0` として登録
        if results is None:
            # solution エイリアス (kwargs) > result 引数 の優先度
            single_result = kwargs.get('solution', result)
            if single_result is not None:
                # 単一ブロックを内部形式へ変換
                variables = single_result.get('variables') if isinstance(single_result, dict) else single_result
                # 旧 API のキー (objective/energy)
                single_result_obj = None
                if isinstance(single_result, dict):
                    single_result_obj = single_result.get('objective') or single_result.get('energy')
                self.results = {
                    'block0': {
                        'solution': variables,
                        'energy': single_result_obj,
                        'device': result_type,
                    }
                }
            else:
                self.results = {}
        else:
            self.results = results

        # 互換用属性
        self.result_type = result_type or self._infer_last_device()

        # `solution` は最初のブロックを参照（存在しない場合は空辞書）
        self.solution = self.get_block_solution(next(iter(self.results)) if self.results else 'block0')

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
    def _set_block(self, block_label: str, variables: Dict[str, Any], objective: Any, **extras):
        """内部ユーティリティ: 1 ブロック分の結果を書き込む。"""
        self.results[block_label] = {
            'solution': variables,
            'energy': objective,
            **extras,
        }
        # 旧 API 属性も更新
        self.solution = variables
        self.result_type = extras.get('device', self.result_type)
        return self

    def from_pulp(self, problem, block_label: str = 'block0'):
        from pulp import value  # local import
        objective_value = value(problem.objective)
        variables = {var.name: var.value() for var in problem.variables()}
        return self._set_block(block_label, variables, objective_value, device='pulp')

    def from_amplify(self, result, block_label: str = 'block0'):
        variables = {str(k): v for k, v in result.best.values.items()}
        return self._set_block(block_label, variables, result.best.objective, device='amplify')

    def from_dimod(self, result, block_label: str = 'block0'):
        return self._set_block(block_label, result.first.sample, result.first.energy, device='dimod')

    def from_scipy(self, result, block_label: str = 'block0'):
        import numpy as np  # noqa: F401 – 型検査用に保持
        variables = {f"q{i}": v for i, v in enumerate(result.x)}
        return self._set_block(block_label, variables, result.fun, device='scipy')

    # ------------------------------------------------------------------
    # to_* 系 (QuDataAnnealingOutput → 外部ライブラリ)
    # ------------------------------------------------------------------
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


# エイリアス（旧クラス名を残しておく）
QdAnnOut = QuDataAnnealingOutput