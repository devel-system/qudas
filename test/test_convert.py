from qudas.convert import dict_to_pyqubo, dict_to_pulp, dict_to_amplify, dict_to_matrix

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