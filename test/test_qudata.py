from amplify import VariableGenerator, Poly
import numpy as np
import pulp
from qudas.qudata import QuData

##############################
### pulp
##############################
# 変数の定義
q0 = pulp.LpVariable('q0', lowBound=0, upBound=1, cat='Binary')
q1 = pulp.LpVariable('q1', lowBound=0, upBound=1, cat='Binary')

# 問題の定義 (2q0-q1)
problem = pulp.LpProblem('QUBO', pulp.LpMinimize)
problem += 2 * q0 - q1

##############################
### amplify
##############################
gen = VariableGenerator()
q = gen.array("Binary", shape=(3))
objective = q[0] * q[1] - q[2]

##############################
### array
##############################
array = np.array([
        [1, 1, 0],
        [0, 2, 0],
        [0, 0, -1],
    ])

if __name__ == '__main__':

    # dict
    qd1 = QuData({('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0})
    print(qd1.prob)

    # pulp
    qd2 = QuData()
    qd2.from_pulp(problem)
    print(qd2.prob)

    # amplify
    qd3 = QuData()
    qd3.from_amplify(objective)
    print(qd3.prob)

    # array
    qd4 = QuData()
    qd4.from_array(array)
    print(qd4.prob)

    # csv
    qd5 = QuData()
    qd5.from_csv(path='./data/qudata.csv')
    print(qd5.prob)

    # json
    qd6 = QuData()
    qd6.from_json(path='./data/qudata.json')
    print(qd6.prob)

    # add
    print(qd1 + qd2)

    # to_pulp
    print(qd2.to_pulp())

    # to_amplify
    print(qd3.to_amplify())

    # to_array
    print(qd4.to_array())