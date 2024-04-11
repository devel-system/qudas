import pulp
from amplify import VariableGenerator, Poly
from pyqubo import Binary, Add
import numpy as np

class Converter:
    def __init__(self, prob=None) -> None:
        self.prob = prob
        self._isinstance()

    def _isinstance(self):
        if isinstance(self.prob, dict):
            self.type = "dict"
        elif isinstance(self.prob, pulp.LpProblem):
            self.type = "pulp"
        elif isinstance(self.prob, Poly):
            self.type = "amplify"
        elif isinstance(self.prob, Add):
            self.type = "pyqubo"
        elif isinstance(self.prob, np.ndarray):
            self.type = "matrix"
        else:
            raise TypeError("対応していない型です。")

    def to_dict(self):
        if self.type == "pulp":
            return self._pulp_to_dict(self.prob)
        elif self.type == "amplify":
            return self._amplify_to_dict(self.prob)
        elif self.type == "pyqubo":
            return self._pyqubo_to_amplify(self.prob)
        # elif self.type == "matrix":
        #     return self._matrix_to_dict(self.prob)
        else:
            raise TypeError("同じ型です。")

    def to_pulp(self):
        if self.type == "dict":
            return self._dict_to_pulp(self.prob)
        elif self.type == "amplify":
            return self._amplify_to_pulp(self.prob)
        elif self.type == "pyqubo":
            return self._pyqubo_to_pulp(self.prob)
        # elif self.type == "matrix":
        #     return self._matrix_to_pulp(self.prob)
        else:
            raise TypeError("同じ型です。")

    def to_amplify(self):
        if self.type == "dict":
            return self._dict_to_amplify(self.prob)
        elif self.type == "pulp":
            return self._pulp_to_amplify(self.prob)
        elif self.type == "pyqubo":
            return self._pyqubo_to_amplify(self.prob)
        # elif self.type == "matrix":
        #     return self._matrix_to_amplify(self.prob)
        else:
            raise TypeError("同じ型です。")

    def to_pyqubo(self):
        if self.type == "dict":
            return self._dict_to_pyqubo(self.prob)
        elif self.type == "pulp":
            return self._pulp_to_pyqubo(self.prob)
        elif self.type == "amplify":
            return self._amplify_to_pyqubo(self.prob)
        # elif self.type == "matrix":
        #     return self._matrix_to_pyqubo(self.prob)
        else:
            raise TypeError("同じ型です。")

    ##############################
    ### pulp
    ##############################
    def _pulp_to_dict(self, prob=pulp.LpProblem) -> dict:
        qubo = {}
        for var in prob.objective.to_dict():
            qubo[(var['name'], var['name'])] = var['value']

        return qubo

    def _pulp_to_amplify(self, prob=pulp.LpProblem) -> Poly:
        gen = VariableGenerator()
        q = gen.array("Binary", len(prob.objective.to_dict()))

        qubo = 0
        for i, var in enumerate(prob.objective.to_dict()):
            qubo += q[i] * var['value']

        return qubo

    def _pulp_to_pyqubo(self, prob=pulp.LpProblem) -> dict:
        qubo = 0
        for var in prob.objective.to_dict():
            qubo += Binary(var['name']) * var['value']

        return qubo

    def _pulp_to_matrix(self, prob=pulp.LpProblem) -> dict:
        qubo = np.zeros((len(prob.objective.to_dict()), len(prob.objective.to_dict())))
        for i, var in enumerate(prob.objective.to_dict()):
            qubo[i][i] = var['value']

        return qubo

    ##############################
    ### amplify
    ##############################
    def _amplify_to_dict(self, prob=Poly) -> dict:
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

        return qubo

    def _amplify_to_pulp(self, prob=Poly) -> pulp.LpProblem:
        q = []
        for variables in prob.variables:
            lp_variable = pulp.LpVariable(variables.name, lowBound=0, upBound=1, cat='Binary')
            q.append(lp_variable)

        qubo = pulp.LpProblem('QUBO', pulp.LpMinimize)
        for key, value in prob.as_dict().items():

            # 1変数
            if len(key) == 1:
                qubo += q[key[0]] * value

            # 2変数以上
            else:
                raise ValueError("pulpは2変数以上に対応していません。")

        return qubo

    def _amplify_to_pyqubo(self, prob=Poly) -> dict:
        variables = prob.variables
        qubo = 0
        for key, value in prob.as_dict().items():
            sub_qubo = 1
            for ki in range(len(key)):
                sub_qubo *= Binary(variables[key[ki]].name)
            qubo += sub_qubo * value

        return qubo

    def _amplify_to_matrix(self, prob=Poly) -> np.ndarray:
        qubo = np.zeros((len(prob.variables), len(prob.variables)))
        for key, value in prob.as_dict().items():

            # 1変数
            if len(key) == 1:
                qubo[key[0], key[0]] = value

            # 2変数
            elif len(key) == 2:
                qubo[key[0], key[1]] = value

            # 3変数以上
            else:
                raise ValueError("matrixは3変数以上に対応していません。")

        return qubo

    ##############################
    ### pyqubo
    ##############################
    def _pyqubo_to_dict(self, prob=Add) -> dict:
        model = prob.compile().to_qubo()

        return model[0]

    def _pyqubo_to_pulp(self, prob=Add) -> pulp.LpProblem:
        model = prob.compile()

        q = []
        for name in model.variables:
            lp_variable = pulp.LpVariable(name, lowBound=0, upBound=1, cat='Binary')
            q.append(lp_variable)

        qubo = pulp.LpProblem('QUBO', pulp.LpMinimize)
        for key, value in model.to_qubo(index_label=True)[0].items():

            # 1変数
            if key[0] == key[1]:
                qubo += q[key[0]] * value

            # 2変数以上
            else:
                raise ValueError("pulpは2変数以上に対応していません。")

        return qubo

    def _pyqubo_to_amplify(self, prob=Add) -> dict:
        model = prob.compile()

        gen = VariableGenerator()
        q = gen.array("Binary", len(model.variables))

        qubo = 0
        for key, value in model.to_qubo(index_label=True)[0].items():
            qubo += q[key[0]] * q[key[1]] * value

        return qubo

    def _pyqubo_to_matrix(self, prob=Add) -> np.ndarray:
        model = prob.compile()

        qubo = np.zeros((len(model.variables), len(model.variables)))
        for key, value in model.to_qubo(index_label=True)[0].items():

            # 1変数
            if len(key) == 1:
                qubo[key[0], key[0]] = value

            # 2変数
            elif len(key) == 2:
                qubo[key[0], key[1]] = value

            # 3変数以上
            else:
                raise ValueError("matrixは3変数以上に対応していません。")

        return qubo

    ##############################
    ### dict
    ##############################
    def _dict_to_pyqubo(self, prob=dict) -> dict:
        variables = []
        for key in prob.keys():
            for k in key:
                variables.append(k)

        variables = list(set(variables))

        q = []
        for variable in variables:
            q.append(Binary(variable))

        qubo = 0
        for key, value in prob.items():
            sub_qubo = 1
            for k in key:
                variable_index = variables.index(k)
                sub_qubo *= q[variable_index]

            qubo += sub_qubo * value

        return qubo

    def _dict_to_pulp(self, prob=dict) -> pulp.LpProblem:
        variables = []
        for key in prob.keys():
            for k in key:
                variables.append(k)

        variables = list(set(variables))

        q = []
        for name in variables:
            lp_variable = pulp.LpVariable(name, lowBound=0, upBound=1, cat='Binary')
            q.append(lp_variable)

        qubo = pulp.LpProblem('QUBO', pulp.LpMinimize)
        for key, value in prob.items():

            # 1変数
            if key[0] == key[1]:
                variable_index = variables.index(key[0])
                qubo += q[variable_index] * value

            # 2変数以上
            else:
                raise ValueError("pulpは2変数以上に対応していません。")

        return qubo

    def _dict_to_amplify(self, prob=dict) -> dict:
        variables = []
        for key in prob.keys():
            for k in key:
                variables.append(k)

        variables = list(set(variables))

        gen = VariableGenerator()
        q = gen.array("Binary", len(variables))

        qubo = 0
        for key, value in prob.items():
            sub_qubo = 1
            for k in key:
                variable_index = variables.index(k)
                sub_qubo *= q[variable_index]

            qubo += sub_qubo * value

        return qubo

    def _dict_to_matrix(self, prob=dict) -> np.ndarray:
        variables = []
        for key in prob.keys():
            for k in key:
                variables.append(k)

        variables = list(set(variables))

        qubo = np.zeros((len(variables), len(variables)))
        for key, value in prob.items():

            # 1変数 or 2変数
            if len(key) == 2:
                variable_index_0 = variables.index(key[0])
                variable_index_1 = variables.index(key[1])
                qubo[variable_index_0, variable_index_1] = value

            # 3変数以上
            else:
                raise ValueError("matrixは3変数以上に対応していません。")

        return qubo


if __name__ == '__main__':

    ##############################
    ### pulp
    ##############################
    # 変数の定義
    q0 = pulp.LpVariable('q0', lowBound=0, upBound=1, cat='Binary')
    q1 = pulp.LpVariable('q1', lowBound=0, upBound=1, cat='Binary')

    # 問題の定義 (q0+q1)
    problem = pulp.LpProblem('QUBO', pulp.LpMinimize)
    # problem += 2 * q0 - q1
    problem += 2 * q0 - q1

    # 解を求める
    problem.solve()
    print(problem)
    # print(q0.value())
    # print(q1.value())

    # 結果
    print(f"{problem.objective.to_dict()=}")
    print(f"{pulp_to_dict(problem)=}")
    print(f"{pulp_to_amplify(problem)=}")
    print(f"{pulp_to_pyqubo(problem)=}")
    print(f"{pulp_to_matrix(problem)=}")

    ##############################
    ### amplify
    ##############################
    gen = VariableGenerator()
    q = gen.array("Binary", shape=(3))
    # objective = q[0] * q[1] * q[2] # 3次元
    # objective = q[0] * q[1] - q[2] # 2次元
    objective = q[0] + q[1] - q[2] # 1次元

    # 結果
    print(f"{objective=}")
    print(f"{amplify_to_dict(objective)=}")     # 2次元まで
    print(f"{amplify_to_pulp(objective)=}")     # 1次元のみ
    print(f"{amplify_to_pyqubo(objective)=}")   # 全てOK
    print(f"{amplify_to_matrix(objective)=}")   # 2次元まで

    # print(objective.as_dict())
    # print(objective.variables)

    ##############################
    ### pyqubo
    ##############################
    # 変数の定義
    q0 = Binary("q0")
    q1 = Binary("q1")
    q2 = Binary("q2")

    # 問題の定義
    # qubo = q0 * q1 * q2
    # qubo = q0 * q1 - q2
    qubo = q0 + q1 - q2
    model = qubo.compile()

    # print(f"{qubo=}")
    # print(f"{type(qubo)=}")
    # print(model.to_qubo())
    # print(model.variables)

    # 結果
    print(f"{qubo=}")
    print(f"{pyqubo_to_dict(qubo)=}")       # 2次元まで
    print(f"{pyqubo_to_pulp(qubo)=}")       # 1次元のみ
    print(f"{pyqubo_to_amplify(qubo)=}")    # 全てOK
    print(f"{pyqubo_to_matrix(qubo)=}")     # 2次元まで

    ##############################
    ### dict
    ##############################
    # prob = {('q0', 'q1', 'q2'): 1.0}
    # prob = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
    prob = {('q0', 'q0'): 1.0, ('q1', 'q1'): 1.0, ('q2', 'q2'): -1.0}

    print(f"{dict_to_pyqubo(prob)=}")       # 全てOK
    print(f"{dict_to_pulp(prob)=}")         # 1次元のみ
    print(f"{dict_to_amplify(prob)=}")      # 全てOK
    print(f"{dict_to_matrix(prob)=}")       # 2次元まで