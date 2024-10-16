from .qudata_base import QuDataBase
from typing import Dict, Any
from pulp import LpVariable, LpProblem, LpMinimize, value
from amplify import Model, Result

# from datetime import timedelta
import dimod
import numpy as np
from scipy.optimize import OptimizeResult


class QuDataOutput(QuDataBase):
    def __init__(self, result: Dict[str, Any] = None, result_type: str = None):
        """
        初期データとして出力データを受け取るクラス。

        Args:
            result (dict, optional): 計算結果データ。デフォルトはNone。
            result_type (str, optional): 結果の形式。デフォルトはNone。
        """
        super().__init__(result)
        self.result = self.data  # dataをresultとして扱う
        self.result_type = result_type

    # PuLPの計算結果を受け取る
    def from_pulp(self, problem: LpProblem) -> "QuDataOutput":
        # 目的関数の値を取得
        objective_value = value(problem.objective)

        # 変数の値を取得
        variables = {var.name: var.value() for var in problem.variables()}
        self.result = {'variables': variables, 'objective': objective_value}
        self.result_type = 'pulp'
        return self

    # Amplifyの計算結果を受け取る
    def from_amplify(self, result: Result) -> "QuDataOutput":
        variables = {str(k): v for k, v in result.best.values.items()}
        self.result = {'variables': variables, 'objective': result.best.objective}
        self.result_type = 'amplify'
        return self

    # Dimodの計算結果を受け取る
    def from_dimod(self, result: dimod.SampleSet) -> "QuDataOutput":
        self.result = {
            'variables': result.first.sample,
            'objective': result.first.energy,
        }
        self.result_type = 'dimod'
        return self

    # SciPyの計算結果を受け取る
    def from_scipy(self, result: OptimizeResult) -> "QuDataOutput":
        variables = {f"q{i}": v for i, v in enumerate(result.x)}
        self.result = {'variables': variables, 'objective': result.fun}
        self.result_type = 'sympy'
        return self

    # Amplify形式に変換（現状は対応不可）
    # def to_amplify(self) -> Result:

    #     # best
    #     best = Result.Solution()
    #     best.values = self.data["variables"]
    #     best.feasible = True
    #     best.objective = self.data["objective"]
    #     best.time = timedelta()

    #     result = Result(
    #         best=best,
    #         client_result=None,
    #         embedding=None,
    #         execution_time=timedelta(),
    #         filter_solution=False,
    #         intermediate=Result.ModelConversion()
    #     )
    #     return result

    # Dimod形式に変換
    def to_dimod(self) -> dimod.SampleSet:
        sampleset = dimod.SampleSet.from_samples(
            samples_like=dimod.as_samples(self.result["variables"]),
            vartype='BINARY',
            energy=self.result["objective"],
        )
        return sampleset

    # SciPy形式に変換
    def to_scipy(self) -> OptimizeResult:
        # 最適化後の結果（例: 最適解とその他の情報を仮定）
        solution = self.result["variables"]  # 手動で得た最適化後の変数
        fun_value = self.result["objective"]  # 目的関数の最小値
        success = True  # 収束したかどうか
        status = 0  # ステータスコード（0は成功）
        message = 'Optimization terminated successfully.'  # 最適化メッセージ
        nfev = 0  # 目的関数を評価した回数
        nit = 0  # 反復回数

        # solution の変数値をリストに変換
        x = np.array(list(solution.values()))

        # OptimizeResult オブジェクトの作成
        result = OptimizeResult(
            x=x,  # 最適化された変数の値
            fun=fun_value,  # 目的関数の最小値
            success=success,  # 成功フラグ
            status=status,  # ステータスコード
            message=message,  # 結果メッセージ
            nfev=nfev,  # 目的関数評価回数
            nit=nit,  # 反復回数
        )
        return result
