from qudas.converter import Converter

##############################
### dict
##############################
# prob = {('q0', 'q1', 'q2'): 1.0}
# prob = {('q0', 'q1'): 1.0, ('q2', 'q2'): -1.0}
prob = {('q0', 'q0'): 1.0, ('q1', 'q1'): 1.0, ('q2', 'q2'): -1.0}
converter = Converter(prob)

print(f"{converter.to_amplify()}")  # 全てOK
print(f"{converter.to_pulp()}")     # 1次元のみ
print(f"{converter.to_pyqubo()=}")  # 全てOK
print(f"{converter.to_dict()}")     # Error