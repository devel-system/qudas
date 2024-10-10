import unittest
from qudas.qudata import QuDataOutput

# その他必要なパッケージ
from amplify import VariableGenerator, Poly
import numpy as np
import pandas as pd
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import dimod
from sympy import Symbol

# 最適化問題の定義（例）
problem = pulp.LpProblem("Example Problem", pulp.LpMinimize)
x = pulp.LpVariable('x', 0, 1)
y = pulp.LpVariable('y', 0, 1)
problem += x + y  # 目的関数
problem += x + 2 * y >= 1  # 制約
problem.solve()

# QudataOutputにPuLPの計算結果をセット
qdata = QuDataOutput()
qdata.from_pulp(problem)

# 他の形式に変換（例）
amplify_result = qdata.to_amplify()
print(amplify_result)