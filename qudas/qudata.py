from amplify import VariableGenerator, Poly
import csv
import json
from pulp import LpProblem, LpVariable, LpMinimize
import dimod
from pyqubo import Binary, Add
import networkx as nx
import numpy as np
import pandas as pd
import sympy

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
            self.prob   = None

        elif isinstance(prob, dict):
            self.prob = prob

        else:
            raise TypeError(f"{type(prob)}は対応していない型です。")

    def __add__(self, other):
        qubo = self.prob.copy()
        for k, v in other.prob.items():
            if k in qubo:
                qubo[k] += v
            else:
                qubo[k] = v

        return QuData(qubo)

    def __sub__(self, other):
        qubo = self.prob.copy()
        for k, v in other.prob.items():
            if k in qubo:
                qubo[k] -= v
            else:
                qubo[k] = -v

        return QuData(qubo)

    def __mul__(self, other):
        qubo = {}
        for k1, v1 in self.prob.items():
            for k2, v2 in other.prob.items():

                # keyのリストを作成
                k = set(k1 + k2)
                for _k in qubo.keys():

                    # 要素が重複する場合
                    if k == set(_k):
                        qubo[_k] += v1 * v2
                        break
                else:

                    # 要素を新規作成
                    if len(k) == 1:
                        qubo[(list(k)[0], list(k)[0])] = v1 * v2
                    else:
                        qubo[tuple(k)] = v1 * v2

        return QuData(qubo)

    def __pow__(self, other: int):
        if isinstance(other, int):
            qudata = self
            for _ in range(other):
                qudata *= qudata

            return qudata
        else:
            raise TypeError(f"{type(other)}は対応していない型です。")

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

        if isinstance(prob, np.ndarray):
            qubo = {}
            for i, ai in enumerate(prob):
                for j, aij in enumerate(ai):
                    if aij == 0:
                        continue

                    if (f"q_{j}", f"q_{i}") in qubo:
                        qubo[(f"q_{j}", f"q_{i}")] += aij
                    else:
                        qubo[(f"q_{i}", f"q_{j}")] = aij

            self.prob = qubo
            return self
        else:
            raise TypeError(f"{type(prob)}は対応していない型です。")

    def from_csv(self, path: str, encoding='utf-8-sig'):
        """csvデータを読み込む

        Args:
            path (str): ファイルパス文字列

        Raises:
            Exception: 形式エラー

        Returns:
            Qudata: 量子データ
        """

        with open(path, encoding=encoding, newline='') as f:
            try:
                qubo = {}
                csvreader = csv.reader(f)
                for i, ai in enumerate(csvreader):
                    for j, aij in enumerate(ai):
                        if float(aij) == 0:
                            continue

                        if (f"q_{j}", f"q_{i}") in qubo:
                            qubo[(f"q_{j}", f"q_{i}")] += float(aij)
                        else:
                            qubo[(f"q_{i}", f"q_{j}")] = float(aij)

                self.prob = qubo
                return self

            except Exception:
                raise "読み取りエラー"

    def from_json(self, path: str):
        """jsonデータを読み込む

        Args:
            path (str): ファイルパス文字列

        Raises:
            Exception: 形式エラー

        Returns:
            Qudata: 量子データ
        """

        with open(path) as f:
            try:
                qubo = {}
                jd = json.load(f)
                for q in jd["qubo"]:
                    qubo[(q["key"][0], q["key"][1])] = q["value"]

                self.prob = qubo
                return self

            except Exception:
                raise "読み取りエラー"

    def from_networkx(self, prob: nx.Graph):
        """グラフデータを読み込む

        Args:
            prob (nx.Graph): networkxのグラフデータ

        Raises:
            TypeError: 形式エラー

        Returns:
            Qudata: 量子データ
        """

        if isinstance(prob, nx.Graph):
            qubo = {}
            for e in prob.edges():
                if (f"q_{e[0]}", f"q_{e[1]}") in qubo:
                    qubo[(f"q_{e[0]}", f"q_{e[1]}")] += 1
                else:
                    qubo[(f"q_{e[0]}", f"q_{e[1]}")] = 1

            self.prob = qubo
            return self
        else:
            raise TypeError(f"{type(prob)}は対応していない型です。")

    def from_pandas(self, prob: pd.DataFrame):
        """pandasデータを読み込む

        Args:
            prob (pd.DataFrame): pandasのデータフレーム

        Raises:
            TypeError: 形式エラー

        Returns:
            Qudata: 量子データ
        """

        if isinstance(prob, pd.DataFrame):
            key1_list = prob.columns.tolist()
            key2_list = prob.index.tolist()

            qubo = {}
            for k1 in key1_list:
                for k2 in key2_list:
                    if prob[k1][k2] == 0:
                        continue

                    if (k1, k2) in qubo:
                        qubo[(k1, k2)] += prob[k1][k2]
                    else:
                        qubo[(k1, k2)] = prob[k1][k2]

            self.prob = qubo
            return self

        else:
            raise TypeError(f"{type(prob)}は対応していない型です。")

    def from_dimod_bqm(self, prob: dimod.BinaryQuadraticModel):
        """dimodのbqmデータを読み込む

        Args:
            prob (dimod.BinaryQuadraticModel): dimodのbqmデータ

        Raises:
            TypeError: 形式エラー

        Returns:
            Qudata: 量子データ
        """

        if isinstance(prob, dimod.BinaryQuadraticModel):
            qubo = dict(prob.quadratic).copy()
            for k, v in prob.linear.items():
                if v == 0:
                    continue

                qubo[(k, k)] = v

            self.prob = qubo
            return self

        else:
            raise TypeError(f"{type(prob)}は対応していない型です。")

    def from_sympy(self, prob: sympy.core.expr.Expr):
        """sympyデータを読み込む

        Args:
            prob (sympy.core.expr.Expr): sympyデータ

        Raises:
            TypeError: 形式エラー

        Returns:
            Qudata: 量子データ
        """

        if isinstance(prob, sympy.core.expr.Expr):
            qubo = {}
            for term in prob.as_ordered_terms():

                # 係数と変数を取得
                v, k = term.as_coeff_mul()

                if len(k) == 1:
                    variable = term.free_symbols
                    qubo[(str(list(variable)[0]), str(list(variable)[0]))] = v
                else:
                    k = tuple([str(_k) for _k in k])
                    qubo[k] = v

            self.prob = qubo
            return self

        else:
            raise TypeError(f"{type(prob)}は対応していない型です。")

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

    def to_csv(self, name="qudata") -> None:
        """numpy形式に変換

        Raises:
            ValueError: 次元エラー
        """

        qubo = self.to_array()
        with open(f"{name}.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerows(qubo)

    def to_json(self, name="qudata") -> None:
        """json形式に変換

        Args:
            name (str, optional): ファイル名. Defaults to "qudata".
        """

        qubo = []
        for key, value in self.prob.items():
            qubo.append({"key": list(key), "value": value})

        with open(f"{name}.json", 'w') as f:
            json.dump(qubo, f, indent=2)

    def to_networkx(self) -> nx.Graph:
        """networkx形式に変換

        Raises:
            ValueError: 次元エラー

        Returns:
            nx.Graph: networkxのグラフデータ
        """

        variables = []
        for key in self.prob.keys():
            for k in key:
                variables.append(k)

        variables = list(set(variables))

        G = nx.Graph()
        for key, value in self.prob.items():

            # 1変数 or 2変数
            if len(key) == 2:
                variable_index_0 = variables.index(key[0])
                variable_index_1 = variables.index(key[1])
                G.add_edge(variable_index_0, variable_index_1, value=value)

            # 3変数以上
            else:
                raise ValueError("networkxは3変数以上に対応していません。")

        return G

    def to_pandas(self) -> pd.DataFrame:
        """pandas形式に変換

        Returns:
            pd.DataFrame: pandasデータ
        """

        variables = []
        for key in self.prob.keys():
            for k in key:
                variables.append(k)

        variables = list(set(variables))

        array = self.to_array()
        df = pd.DataFrame(array,
                          columns=variables,
                          index=variables)

        return df

    def to_dimod_bqm(self) -> dimod.BinaryQuadraticModel:
        """dimodのbqm形式に変換

        Returns:
            dimod.BinaryQuadraticModel: dimodのbqmデータ
        """

        linear = {}
        quadratic = {}
        for k, v in self.prob.items():
            if k[0] == k[1]:
                linear[k[0]] = v
            else:
                quadratic[(k[0], k[1])] = v

        bqm = dimod.BinaryQuadraticModel(linear, quadratic, vartype='BINARY')

        return bqm

    def to_sympy(self) -> sympy.core.expr.Expr:
        """sympy形式に変換

        Returns:
            sympy.core.expr.Expr: sympyの多項式データ
        """

        sympy_prob = 0
        for k, v in self.prob.items():
            if k[0] == k[1]:
                sympy_prob += sympy.Symbol(k[0]) * v
            else:
                var1 = sympy.Symbol(k[0])
                var2 = sympy.Symbol(k[1])
                sympy_prob += var1 * var2 * v

        return sympy_prob
