from collections.abc import Mapping


class QdAnnealingIR(Mapping):
    """量子アニーリング用の中間表現 (QUBO) を表すクラス。

    旧 `QuDataInput` で担っていた QUBO 変換・演算機能を移植した。
    `qubo` は dict で保持し、キーは変数名のタプル、値は係数。
    """

    # ------------------------------
    # コンストラクタ & 演算子オーバーロード
    # ------------------------------
    from typing import Optional  # type: ignore

    def __init__(self, qubo: Optional[dict] = None):
        self.qubo: dict = {}
        if qubo is None:
            self.qubo = {}
        elif isinstance(qubo, dict):
            self.qubo = qubo
        else:
            raise TypeError(f"{type(qubo)}は対応していない型です。")

    # 辞書の比較でキー順を無視する際に使うユーティリティ
    @staticmethod
    def _merge_dict(a: dict, b: dict, op):
        """2つの QUBO dict を merge し、値は op で結合"""
        result = a.copy()
        for k, v in b.items():
            if k in result:
                result[k] = op(result[k], v)
            else:
                result[k] = op(0, v)
        return result

    def __add__(self, other: "QdAnnealingIR") -> "QdAnnealingIR":
        return QdAnnealingIR(
            self._merge_dict(self.qubo, other.qubo, lambda x, y: x + y)
        )

    def __sub__(self, other: "QdAnnealingIR") -> "QdAnnealingIR":
        return QdAnnealingIR(
            self._merge_dict(self.qubo, other.qubo, lambda x, y: x - y)
        )

    def __mul__(self, other: "QdAnnealingIR") -> "QdAnnealingIR":
        qubo = {}
        for k1, v1 in self.qubo.items():
            for k2, v2 in other.qubo.items():
                key_set = set(k1 + k2)
                # 既に存在するkeyがあるか確認
                found = False
                for _k in qubo.keys():
                    if key_set == set(_k):
                        qubo[_k] += v1 * v2
                        found = True
                        break
                if not found:
                    if len(key_set) == 1:
                        var = list(key_set)[0]
                        qubo[(var, var)] = v1 * v2
                    else:
                        qubo[tuple(key_set)] = v1 * v2
        return QdAnnealingIR(qubo)

    def __pow__(self, other: int) -> "QdAnnealingIR":
        if isinstance(other, int):
            result = QdAnnealingIR(self.qubo)
            for _ in range(1, other):
                result = result * self
            return result
        raise TypeError(f"{type(other)}は対応していない型です。")

    # ------------------------------
    # 変換 (from_*)
    # ------------------------------
    # 外部ライブラリ読み込み
    from typing import Any, Dict, Optional
    import csv, json
    import numpy as np
    import networkx as nx
    import pandas as pd
    import sympy
    import dimod
    from amplify import Poly
    from pulp import LpProblem
    from pyqubo import Base

    def from_pulp(self, prob: "LpProblem") -> "QdAnnealingIR":  # type: ignore[name-defined]
        from pulp import LpProblem  # local import to avoid heavy dep if未使用

        if isinstance(prob, LpProblem):
            qubo = {}
            for var in prob.objective.to_dict():
                qubo[(var["name"], var["name"])] = var["value"]
            self.qubo = qubo
            return self
        raise TypeError(f"{type(prob)}は対応していない型です。")

    def from_amplify(self, prob: "Poly") -> "QdAnnealingIR":  # type: ignore[name-defined]
        from amplify import Poly  # type: ignore

        if isinstance(prob, Poly):
            variables = prob.variables
            qubo: dict = {}
            for key, value in prob.as_dict().items():
                if len(key) == 0:
                    continue  # 定数
                elif len(key) == 1:
                    qubo[(variables[key[0]].name, variables[key[0]].name)] = value
                elif len(key) == 2:
                    qubo[(variables[key[0]].name, variables[key[1]].name)] = value
                else:
                    raise ValueError("3変数以上は対応していません。")
            self.qubo = qubo
            return self
        raise TypeError(f"{type(prob)}は対応していない型です。")

    def from_pyqubo(self, prob: "Base") -> "QdAnnealingIR":  # type: ignore[name-defined]
        from pyqubo import Base  # type: ignore

        if isinstance(prob, Base):
            qubo = prob.compile().to_qubo()
            self.qubo = qubo[0]
            return self
        raise TypeError(f"{type(prob)}は対応していない型です。")

    def from_array(self, prob: "np.ndarray") -> "QdAnnealingIR":  # type: ignore[name-defined]
        import numpy as np  # noqa

        if isinstance(prob, np.ndarray):
            qubo: dict = {}
            for i, ai in enumerate(prob):
                for j, aij in enumerate(ai):
                    if aij == 0:
                        continue
                    if (f"q_{j}", f"q_{i}") in qubo:
                        qubo[(f"q_{j}", f"q_{i}")] += aij
                    else:
                        qubo[(f"q_{i}", f"q_{j}")] = aij
            self.qubo = qubo
            return self
        raise TypeError(f"{type(prob)}は対応していない型です。")

    def from_csv(self, path: str, encoding: str = "utf-8-sig") -> "QdAnnealingIR":
        import csv  # local import

        try:
            with open(path, encoding=encoding, newline="") as f:
                qubo: dict = {}
                csvreader = csv.reader(f)
                for i, ai in enumerate(csvreader):
                    for j, aij in enumerate(ai):
                        if float(aij) == 0:
                            continue
                        if (f"q_{j}", f"q_{i}") in qubo:
                            qubo[(f"q_{j}", f"q_{i}")] += float(aij)
                        else:
                            qubo[(f"q_{i}", f"q_{j}")] = float(aij)
                self.qubo = qubo
                return self
        except Exception as e:
            raise ValueError("読み取りエラー") from e

    def from_json(self, path: str) -> "QdAnnealingIR":
        import json  # local import

        try:
            with open(path) as f:
                qubo: dict = {}
                jd = json.load(f)
                for q in jd["qubo"]:
                    qubo[(q["key"][0], q["key"][1])] = q["value"]
                self.qubo = qubo
                return self
        except Exception as e:
            raise ValueError("読み取りエラー") from e

    def from_networkx(self, prob: "nx.Graph") -> "QdAnnealingIR":  # type: ignore[name-defined]
        import networkx as nx  # noqa

        if isinstance(prob, nx.Graph):
            qubo: dict = {}
            for e in prob.edges():
                if (f"q_{e[0]}", f"q_{e[1]}") in qubo:
                    qubo[(f"q_{e[0]}", f"q_{e[1]}")] += 1
                else:
                    qubo[(f"q_{e[0]}", f"q_{e[1]}")] = 1
            self.qubo = qubo
            return self
        raise TypeError(f"{type(prob)}は対応していない型です。")

    def from_pandas(self, prob: "pd.DataFrame") -> "QdAnnealingIR":  # type: ignore[name-defined]
        import pandas as pd  # noqa

        if isinstance(prob, pd.DataFrame):
            key1_list = prob.columns.tolist()
            key2_list = prob.index.tolist()
            qubo: dict = {}
            for k1 in key1_list:
                for k2 in key2_list:
                    if prob[k1][k2] == 0:
                        continue
                    if (k1, k2) in qubo:
                        qubo[(k1, k2)] += prob[k1][k2]
                    else:
                        qubo[(k1, k2)] = prob[k1][k2]
            self.qubo = qubo
            return self
        raise TypeError(f"{type(prob)}は対応していない型です。")

    def from_dimod_bqm(self, prob: "dimod.BinaryQuadraticModel") -> "QdAnnealingIR":  # type: ignore[name-defined]
        import dimod  # noqa

        if isinstance(prob, dimod.BinaryQuadraticModel):
            qubo = dict(prob.quadratic).copy()
            for k, v in prob.linear.items():
                if v == 0:
                    continue
                qubo[(k, k)] = v
            self.qubo = qubo
            return self
        raise TypeError(f"{type(prob)}は対応していない型です。")

    def from_sympy(self, prob: "sympy.core.expr.Expr") -> "QdAnnealingIR":  # type: ignore[name-defined]
        import sympy  # noqa

        if isinstance(prob, sympy.core.expr.Expr):
            qubo: dict = {}
            for term in prob.as_ordered_terms():
                v, k = term.as_coeff_mul()
                if len(k) == 1:
                    variable = term.free_symbols
                    qubo[(str(list(variable)[0]), str(list(variable)[0]))] = v
                else:
                    k_tuple = tuple([str(_k) for _k in k])
                    qubo[k_tuple] = v
            self.qubo = qubo
            return self
        raise TypeError(f"{type(prob)}は対応していない型です。")

    # ------------------------------
    # 変換 (to_*)
    # ------------------------------
    def to_pulp(self):
        from pulp import LpVariable, LpProblem, LpMinimize  # local import

        variables = list(set(k for key in self.qubo.keys() for k in key))
        q = [
            LpVariable(name, lowBound=0, upBound=1, cat='Binary') for name in variables
        ]
        qubo_prob = LpProblem('QUBO', LpMinimize)
        _qubo = 0
        for key, value in self.qubo.items():
            if key[0] == key[1]:
                variable_index = variables.index(key[0])
                _qubo += q[variable_index] * value
            else:
                raise ValueError("pulpは2変数以上に対応していません。")
        qubo_prob += _qubo
        return qubo_prob

    def to_amplify(self):
        from amplify import VariableGenerator  # type: ignore

        variables = sorted(set(k for key in self.qubo.keys() for k in key))

        gen = VariableGenerator()
        labeled_q = {
            str(name): gen.scalar("Binary", name=str(name)) for name in variables
        }

        qubo_poly = 0
        for key, value in self.qubo.items():
            sub_qubo = 1
            for k in key:
                sub_qubo *= labeled_q[k]
            qubo_poly += sub_qubo * value
        return qubo_poly

    def to_pyqubo(self):
        from pyqubo import Binary  # type: ignore

        variables = list(set(k for key in self.qubo.keys() for k in key))
        q = [Binary(str(variable)) for variable in variables]
        qubo_expr = 0
        for key, value in self.qubo.items():
            sub_qubo = 1
            for k in key:
                variable_index = variables.index(k)
                sub_qubo *= q[variable_index]
            qubo_expr += sub_qubo * value
        return qubo_expr

    def to_array(self):
        import numpy as np  # noqa

        variables = sorted(list(set(k for key in self.qubo.keys() for k in key)))
        qubo_arr = np.zeros((len(variables), len(variables)))
        for key, value in self.qubo.items():
            if len(key) == 2:
                variable_index_0 = variables.index(key[0])
                variable_index_1 = variables.index(key[1])
                qubo_arr[variable_index_0, variable_index_1] = value
            else:
                raise ValueError("matrixは3変数以上に対応していません。")
        return qubo_arr

    def to_csv(self, name: str = "qudata") -> None:
        import csv  # noqa

        qubo_arr = self.to_array()
        try:
            with open(f"{name}.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(qubo_arr)
        except Exception as e:
            raise ValueError("書き出しエラー") from e

    def to_json(self, name: str = "qudata") -> None:
        import json  # noqa

        qubo_json = [
            {"key": list(key), "value": value} for key, value in self.qubo.items()
        ]
        try:
            with open(f"{name}.json", 'w') as f:
                json.dump({"qubo": qubo_json}, f, indent=4)
        except Exception as e:
            raise ValueError("書き出しエラー") from e

    def to_networkx(self):
        import networkx as nx  # noqa

        variables = list(set(k for key in self.qubo.keys() for k in key))
        G = nx.Graph()
        G.add_nodes_from(variables)
        for key, value in self.qubo.items():
            if len(key) == 2 and key[0] != key[1]:
                G.add_edge(key[0], key[1], weight=value)
        return G

    def to_pandas(self):
        import pandas as pd  # noqa

        variables = sorted(set(k for key in self.qubo.keys() for k in key))
        df = pd.DataFrame(0.0, index=variables, columns=variables)
        for key, value in self.qubo.items():
            df.loc[key[0], key[1]] = value
        return df

    def to_dimod_bqm(self):
        import dimod  # noqa

        linear = {}
        quadratic = {}
        for key, value in self.qubo.items():
            if key[0] == key[1]:
                linear[key[0]] = value
            else:
                quadratic[(key[0], key[1])] = value
        return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype='BINARY')

    def to_sympy(self):
        import sympy  # noqa

        expr = 0
        for (i, j), value in self.qubo.items():
            if i == j:
                # 対角項（一次項として扱う）
                expr += sympy.Symbol(i) * value
            else:
                # 非対角項（二次項）
                expr += sympy.Symbol(i) * sympy.Symbol(j) * value
        return expr

    # ------------------------------
    # ユーティリティ
    # ------------------------------
    def to_dict(self) -> dict:
        """簡易ダンプ"""
        return self.qubo

    # alias for old name
    @classmethod
    def from_dict(cls, data: dict) -> "QdAnnealingIR":
        return cls(qubo=data)

    # 旧 API との互換性: qubo プロパティを prob にも提供
    @property
    def prob(self):
        return self.qubo

    # 比較用
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QdAnnealingIR):
            return False
        return self.qubo == other.qubo

    # Mapping インターフェース
    def __len__(self):
        return len(self.qubo)

    def __iter__(self):
        return iter(self.qubo)

    def __getitem__(self, key):
        return self.qubo[key]


QdAnnIR = QdAnnealingIR
