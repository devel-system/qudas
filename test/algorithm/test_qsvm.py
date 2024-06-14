# qudas & pipline
from sklearn.base import TransformerMixin
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from qudas.pipeline import Pipeline

# module
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator

class qSVMKernel(TransformerMixin):
    """qSVMのカーネル計算処理

    Args:
        TransformerMixin: qudasのデータ変換用mixinクラス
    """

    def __init__(self):
        pass

    def set_grobal_params(self, params) -> None:
        self.params = params

    def get_grobal_params(self) -> dict:
        return self.params

    def transform(self, X) -> None:

        zz_feature_map = self.get_grobal_params()["zz_feature_map"]
        y = self.get_grobal_params()["y"]

        if y is None:
            y = X

        sim = AerSimulator()

        x_matrix, y_matrix = [], []
        for x0, x1 in X:
            param0, param1 = zz_feature_map.parameters
            qc = zz_feature_map.assign_parameters({ param0: x0, param1: x1 })
            # .decompose() せずに .save_statevector() を使うとエラーになる。
            qc = qc.decompose()
            qc.save_statevector()
            sv = sim.run(qc).result().get_statevector()
            x_matrix.append(list(np.array(sv)))

        for y0, y1 in y:
            param0, param1 = zz_feature_map.parameters
            qc = zz_feature_map.assign_parameters({ param0: y0, param1: y1 })
            qc = qc.decompose()
            qc.save_statevector()
            sv = sim.run(qc).result().get_statevector()
            y_matrix.append(list(np.array(sv)))

        x_matrix, y_matrix = np.array(x_matrix), np.array(y_matrix)

        kernel = np.abs(
            y_matrix.conjugate() @ x_matrix.transpose()
        )**2

        return kernel

class qSVC(SVC):
    """qSVMのカーネル学習処理

    Args:
        SVC: sklearnのSVCクラス
    """

    def set_grobal_params(self, params) -> None:
        self.params = params

    def get_grobal_params(self) -> dict:
        return self.params

if __name__ == '__main__':
    X, Y = make_circles(n_samples=200, noise=0.05, factor=0.4)

    A = X[np.where(Y==0)]
    B = X[np.where(Y==1)]

    A_label = np.zeros(A.shape[0], dtype=int)
    B_label = np.ones(B.shape[0], dtype=int)

    def make_train_test_sets(test_ratio=.3):
        def split(arr, test_ratio):
            sep = int(arr.shape[0]*(1-test_ratio))
            return arr[:sep], arr[sep:]

        A_label = np.zeros(A.shape[0], dtype=int)
        B_label = np.ones(B.shape[0], dtype=int)
        A_train, A_test = split(A, test_ratio)
        B_train, B_test = split(B, test_ratio)
        A_train_label, A_test_label = split(A_label, test_ratio)
        B_train_label, B_test_label = split(B_label, test_ratio)
        X_train = np.concatenate([A_train, B_train])
        y_train = np.concatenate([A_train_label, B_train_label])
        X_test = np.concatenate([A_test, B_test])
        y_test = np.concatenate([A_test_label, B_test_label])
        return X_train, y_train, X_test, y_test

    train_data, train_labels, test_data, test_labels = make_train_test_sets()

    # 初期パラメータ
    zz_feature_map = ZZFeatureMap(feature_dimension=2, reps=2)
    parameters = { 'zz_feature_map': zz_feature_map, 'y': None }

    # pipeline
    steps = [
        ('qSVMKernel', qSVMKernel()),
        ('SVC', qSVC(kernel='precomputed'))
    ]

    pipe = Pipeline(steps)
    pipe.set_grobal_params(parameters)

    # 学習
    pipe.fit(X=train_data, y=train_labels)

    # 推論
    parameters['y'] = test_data
    pipe.set_grobal_params(parameters)
    pred = pipe.predict(X=train_data)

    # 描画
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax1.set_aspect('equal')
    ax1.set_title("Train Data")
    ax1.set_ylim(-2, 2)
    ax1.set_xlim(-2, 2)
    for (x, y), train_label in zip(train_data, train_labels):
        c = 'C0' if train_label == 0 else 'C3'
        ax1.add_patch(matplotlib.patches.Circle((x, y), radius=.01,
                    fill=True, linestyle='solid', linewidth=4.0,
                    color=c))
    ax1.grid()

    ax2 = fig.add_subplot(122)
    ax2.set_aspect('equal')
    ax2.set_title("Test Data Classification")
    ax2.set_ylim(-2, 2)
    ax2.set_xlim(-2, 2)
    for (x, y), pred_label in zip(test_data, pred):
        c = 'C0' if pred_label == 0 else 'C3'
        ax2.add_patch(matplotlib.patches.Circle((x, y), radius=.01,
                    fill=True, linestyle='solid', linewidth=4.0,
                    color=c))
    ax2.grid()
    plt.show()