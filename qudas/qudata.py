from amplify import VariableGenerator, Poly
from pulp import LpProblem, LpVariable, LpMinimize
from pyqubo import Binary, Add
import numpy as np

class QuData:
    """量子データ"""

    def __init__(self, prob: dict=None) -> None:
        """初期データ（dictデータ）

        Args:
            prob (dict, optional): 最適化問題。デフォルトはNone。

        Raises:
            TypeError: 形式エラー
        """

        if prob is None:
            self.dtype  = None
            self.prob   = None

        elif isinstance(prob, dict):
            self.dtype = "dict"
            self.prob = prob

        else:
            raise TypeError(f"{type(prob)}は対応していない型です。")

    def from_pulp(self, prob: LpProblem):
        """pulpデータを読み込む

        Args:
            prob (LpProblem): 線形計画問題

        Raises:
            TypeError: 形式エラー

        Returns:
            Qudata: 量子データ
        """
        if isinstance(prob, LpProblem):

            qubo = {}
            for var in prob.objective.to_dict():
                qubo[(var['name'], var['name'])] = var['value']

            self.dtype = "pulp"
            self.prob = qubo
            return self
        else:
            raise TypeError(f"{type(prob)}は対応していない型です。")

    def from_amplify(self, prob: Poly):
        """amplifyデータを読み込む

        Args:
            prob (Poly): 組み合わせ最適化問題

        Raises:
            TypeError: 形式エラー

        Returns:
            Qudata: 量子データ
        """
        if isinstance(prob, Poly):
            variables = prob.variables
            qubo = {}
            for key, value in prob.as_dict().items():

                # 1変数
                if len(key) == 1:
                    qubo[(variables[key[0]].name, variables[key[0]].name)] = value

                # 2変数
                elif len(key) == 2:
                    qubo[(variables[key[0]].name, variables[key[1]].name)] = value

                # 3変数以上
                else:
                    raise ValueError("dictは3変数以上に対応していません。")

            self.dtype = "amplify"
            self.prob = qubo
            return self
        else:
            raise TypeError(f"{type(prob)}は対応していない型です。")

    def from_pyqubo(self, prob: Add):
        """pyquboデータを読み込む

        Args:
            prob (Add): 組み合わせ最適化問題

        Raises:
            TypeError: 形式エラー

        Returns:
            Qudata: 量子データ
        """
        if isinstance(prob, Add):
            qubo = prob.compile().to_qubo()
            self.dtype = "pyqubo"
            self.prob = qubo[0]
            return self
        else:
            raise TypeError(f"{type(prob)}は対応していない型です。")

    def from_array(self, prob: np.ndarray):
        """numpyデータを読み込む

        Args:
            prob (np.ndarray): 組み合わせ最適化問題

        Raises:
            TypeError: 形式エラー

        Returns:
            Qudata: 量子データ
        """

        # 開発中
        # if isinstance(self.prob, np.ndarray):
        #     self.dtype = "array"
        #     self.prob = prob
        #     return self
        # else:
        #     raise TypeError(f"{type(prob)}は対応していない型です。")
        pass

    def to_pulp(self) -> LpProblem:
        """pulp形式に変換

        Raises:
            ValueError: 変数エラー

        Returns:
            LpProblem: 線形計画問題
        """

        variables = []
        for key in self.prob.keys():
            for k in key:
                variables.append(k)

        variables = list(set(variables))

        q = []
        for name in variables:
            lp_variable = LpVariable(name, lowBound=0, upBound=1, cat='Binary')
            q.append(lp_variable)

        qubo = LpProblem('QUBO', LpMinimize)
        for key, value in self.prob.items():

            # 1変数
            if key[0] == key[1]:
                variable_index = variables.index(key[0])
                qubo += q[variable_index] * value

            # 2変数以上
            else:
                raise ValueError("pulpは2変数以上に対応していません。")

        return qubo

    def to_amplify(self) -> Poly:
        """amplify形式に変換

        Returns:
            Poly: 組み合わせ最適化問題
        """
        variables = []
        for key in self.prob.keys():
            for k in key:
                variables.append(k)

        variables = list(set(variables))

        gen = VariableGenerator()
        q = gen.array("Binary", len(variables))

        qubo = 0
        for key, value in self.prob.items():
            sub_qubo = 1
            for k in key:
                variable_index = variables.index(k)
                sub_qubo *= q[variable_index]

            qubo += sub_qubo * value

        return qubo

    def to_pyqubo(self) -> Add:
        """pyqubo形式に変換

        Returns:
            Add: 組み合わせ最適化問題
        """
        variables = []
        for key in self.prob.keys():
            for k in key:
                variables.append(k)

        variables = list(set(variables))

        q = []
        for variable in variables:
            q.append(Binary(variable))

        qubo = 0
        for key, value in self.prob.items():
            sub_qubo = 1
            for k in key:
                variable_index = variables.index(k)
                sub_qubo *= q[variable_index]

            qubo += sub_qubo * value

        return qubo

    def to_array(self) -> np.ndarray:
        """numpy形式に変換

        Raises:
            ValueError: 次元エラー

        Returns:
            np.ndarray: QUBO行列
        """
        variables = []
        for key in self.prob.keys():
            for k in key:
                variables.append(k)

        variables = list(set(variables))

        qubo = np.zeros((len(variables), len(variables)))
        for key, value in self.prob.items():

            # 1変数 or 2変数
            if len(key) == 2:
                variable_index_0 = variables.index(key[0])
                variable_index_1 = variables.index(key[1])
                qubo[variable_index_0, variable_index_1] = value

            # 3変数以上
            else:
                raise ValueError("matrixは3変数以上に対応していません。")

        return qubo