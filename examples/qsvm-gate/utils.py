from sklearn.datasets import make_circles
from qiskit_aer import AerSimulator
import numpy as np

X, Y = make_circles(n_samples=200, noise=0.05, factor=0.4)

A = X[np.where(Y == 0)]
B = X[np.where(Y == 1)]

A_label = np.zeros(A.shape[0], dtype=int)
B_label = np.ones(B.shape[0], dtype=int)

def _split(arr, test_ratio):
    sep = int(arr.shape[0] * (1 - test_ratio))
    return arr[:sep], arr[sep:]

def make_train_test_sets(test_ratio=0.3):
    A_label = np.zeros(A.shape[0], dtype=int)
    B_label = np.ones(B.shape[0], dtype=int)
    A_train, A_test = _split(A, test_ratio)
    B_train, B_test = _split(B, test_ratio)
    A_train_label, A_test_label = _split(A_label, test_ratio)
    B_train_label, B_test_label = _split(B_label, test_ratio)
    X_train = np.concatenate([A_train, B_train])
    y_train = np.concatenate([A_train_label, B_train_label])
    X_test = np.concatenate([A_test, B_test])
    y_test = np.concatenate([A_test_label, B_test_label])
    return X_train, y_train, X_test, y_test

def calculate_kernel(zz_feature_map, x_data, y_data=None):
    if y_data is None:
        y_data = x_data

    sim = AerSimulator()
    x_matrix, y_matrix = [], []
    for x0, x1 in x_data:
        param0, param1 = zz_feature_map.parameters
        qc = zz_feature_map.assign_parameters({param0: x0, param1: x1})
        # .decompose() せずに .save_statevector() を使うとエラーになる。
        qc = qc.decompose()
        qc.save_statevector()
        sv = sim.run(qc).result().get_statevector()
        x_matrix.append(list(np.array(sv)))

    for y0, y1 in y_data:
        param0, param1 = zz_feature_map.parameters
        qc = zz_feature_map.assign_parameters({param0: y0, param1: y1})
        qc = qc.decompose()
        qc.save_statevector()
        sv = sim.run(qc).result().get_statevector()
        y_matrix.append(list(np.array(sv)))

    x_matrix, y_matrix = np.array(x_matrix), np.array(y_matrix)

    kernel = np.abs(y_matrix.conjugate() @ x_matrix.transpose()) ** 2

    return kernel