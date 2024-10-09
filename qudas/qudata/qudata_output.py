from .qudata_base import QuDataBase
from typing import Dict, Any
from pulp import LpVariable, LpProblem, LpMinimize, value

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
    def from_pulp(self, problem: LpProblem):
        # 目的関数の値を取得
        objective_value = value(problem.objective)
        # 変数の値を取得
        variables = {var.name: var.value() for var in problem.variables()}
        self.result = {
            'variables': variables,
            'objective': objective_value
        }
        self.result_type = 'pulp'

    # Amplifyの計算結果を受け取る
    def from_amplify(self, result: Dict[str, Any]):
        self.result = result
        self.result_type = 'amplify'

    # Dimodの計算結果を受け取る
    def from_dimod(self, result: Dict[str, Any]):
        self.result = result
        self.result_type = 'dimod'

    # SymPyの計算結果を受け取る
    def from_sympy(self, result: Dict[str, Any]):
        self.result = result
        self.result_type = 'sympy'

    # PuLP形式に変換
    def to_pulp(self) -> Dict[str, Any]:
        if self.result_type == 'pulp':
            return self.result
        else:
            return {'variables': self.result['variables'], 'objective': self.result['objective']}

    # Amplify形式に変換
    def to_amplify(self) -> Dict[str, Any]:
        if self.result_type == 'amplify':
            return self.result
        else:
            binary_matrix = [1 if val > 0 else 0 for val in self.result['variables'].values()]
            return {'variables': binary_matrix, 'objective': self.result['objective']}

    # Dimod形式に変換
    def to_dimod(self) -> Dict[str, Any]:
        if self.result_type == 'dimod':
            return self.result
        else:
            # 仮のサンプルとエネルギーを用いてDimod形式に変換
            samples = [dict(enumerate([int(val > 0) for val in self.result['variables'].values()]))]
            energy = [self.result['objective']]
            return {'variables': samples[0], 'objective': energy[0]}

    # SymPy形式に変換
    def to_sympy(self) -> Dict[str, Any]:
        if self.result_type == 'sympy':
            return self.result
        else:
            solutions = [float(val) for val in self.result['variables'].values()]
            return {'variables': solutions, 'objective': self.result['objective']}

if __name__ == '__main__':
    # 使用例
    from pulp import LpVariable, LpProblem, LpMinimize

    # 最適化問題の定義（例）
    problem = LpProblem("Example Problem", LpMinimize)
    x = LpVariable('x', 0, 1)
    y = LpVariable('y', 0, 1)
    problem += x + y  # 目的関数
    problem += x + 2 * y >= 1  # 制約
    problem.solve()

    # QudataOutputにPuLPの計算結果をセット
    qdata = QudataOutput()
    qdata.from_pulp(problem)

    # 他の形式に変換（例）
    amplify_result = qdata.to_amplify()
    print(amplify_result)

    sympy_result = qdata.to_sympy()
    print(sympy_result)
