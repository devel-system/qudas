# 参考URL
# https://zenn.dev/derwind/articles/dwd-qsvm-qiskit

# qudas & pipline
from qudas.pipeline import Pipeline
from utils import make_train_test_sets, calculate_kernel

# module
from sklearn.svm import SVC
from qiskit.circuit.library import ZZFeatureMap
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # データの取得
    train_data, train_labels, test_data, _ = make_train_test_sets()

    # 初期パラメータ
    zz_feature_map = ZZFeatureMap(feature_dimension=2, reps=2)
    train_kernel = calculate_kernel(zz_feature_map, train_data)

    # pipeline
    steps = [('qSVC', SVC(kernel='precomputed'))]

    pipeline = Pipeline(steps)

    # 学習
    pipeline.fit(X=train_kernel, y=train_labels)

    # 推論
    test_kernel = calculate_kernel(zz_feature_map, train_data, test_data)
    results = pipeline.predict(X=test_kernel)
    pred = results["qSVC"]

    # 描画
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax1.set_aspect('equal')
    ax1.set_title("Train Data")
    ax1.set_ylim(-2, 2)
    ax1.set_xlim(-2, 2)
    for (x, y), train_label in zip(train_data, train_labels):
        c = 'C0' if train_label == 0 else 'C3'
        ax1.add_patch(
            matplotlib.patches.Circle(
                (x, y),
                radius=0.01,
                fill=True,
                linestyle='solid',
                linewidth=4.0,
                color=c,
            )
        )
    ax1.grid()

    ax2 = fig.add_subplot(122)
    ax2.set_aspect('equal')
    ax2.set_title("Test Data Classification")
    ax2.set_ylim(-2, 2)
    ax2.set_xlim(-2, 2)
    for (x, y), pred_label in zip(test_data, pred):
        c = 'C0' if pred_label == 0 else 'C3'
        ax2.add_patch(
            matplotlib.patches.Circle(
                (x, y),
                radius=0.01,
                fill=True,
                linestyle='solid',
                linewidth=4.0,
                color=c,
            )
        )
    ax2.grid()
    plt.show()
