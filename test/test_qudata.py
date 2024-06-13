from amplify import VariableGenerator, Poly
import numpy as np
import pulp
import networkx as nx
import matplotlib.pyplot as plt
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

##############################
### networkx
##############################
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (0, 2)])
# nx.draw_networkx(G)
# plt.show()

if __name__ == '__main__':

    # dict
    qd1 = QuData({('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0})
    print(f"dict={qd1.prob}")

    # pulp
    qd2 = QuData()
    qd2.from_pulp(problem)
    print(f"pulp={qd2.prob}")

    # amplify
    qd3 = QuData()
    qd3.from_amplify(objective)
    print(f"amplify={qd3.prob}")

    # array
    qd4 = QuData()
    qd4.from_array(array)
    print(f"array={qd4.prob}")

    # csv
    qd5 = QuData()
    qd5.from_csv(path='./data/qudata.csv')
    print(f"csv={qd5.prob}")

    # json
    qd6 = QuData()
    qd6.from_json(path='./data/qudata.json')
    print(f"json={qd6.prob}")

    # networkx
    qd7 = QuData()
    qd7.from_networkx(G)
    print(f"networkx={qd7.prob}")

    # add
    print(f"add={qd1 + qd2}")

    # to_pulp
    print(qd2.to_pulp())

    # to_amplify
    print(qd3.to_amplify())

    # to_array
    print(qd4.to_array())

    # to_csv
    qd5.to_csv(name="to-csv")

    # to_json
    qd6.to_json(name="to-json")