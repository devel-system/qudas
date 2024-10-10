# module
import numpy as np
from typing import Callable

# 乱数シードの固定
seed = 1234
rng = np.random.default_rng(seed)

def make_blackbox_func(d: int) -> Callable[[np.ndarray], float]:
    """入力が長さ d のバリナリ値のベクトルで出力が float であるような関数を返却する

    Args:
        d (int): 入力サイズ

    Returns:
        Callable[[np.ndarray], float]: ブラックボックス関数
    """

    # Qを作成 (ブラックボックス)
    rng = np.random.default_rng(seed)
    Q = rng.random((d, d))
    Q = (Q + Q.T) / 2
    Q = Q - np.mean(Q)

    # ブラックボックス関数を作成
    def blackbox(x: np.ndarray) -> float:
        assert x.shape == (d,)  # x は要素数 d の一次元配列
        return x @ Q @ x  # type: ignore

    return blackbox

def init_training_data(d: int, n0: int, blackbox):
    """n0 組の初期教師データを作成する"""
    assert n0 < 2**d

    # n0 個の 長さ d の入力値を乱数を用いて作成
    x = rng.choice(np.array([0, 1]), size=(n0, d))

    # 入力値の重複が発生していたらランダムに値を変更して回避する
    x = np.unique(x, axis=0)
    while x.shape[0] != n0:
        x = np.vstack((x, np.random.randint(0, 2, size=(n0 - x.shape[0], d))))
        x = np.unique(x, axis=0)

    # blackbox 関数を評価して入力値に対応する n0 個の出力を得る
    y = np.zeros(n0)
    for i in range(n0):
        y[i] = blackbox(x[i])

    return x, y