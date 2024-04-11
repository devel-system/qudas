# qudas & pipline
from sklearn.base import BaseEstimator, TransformerMixin
from qudas.base import OptimizerMixin, IteratorMixin
from qudas.pipeline import Pipeline

# module
from amplify import VariableGenerator, Model, FixstarsClient, solve, Poly
import copy
from datetime import timedelta
import numpy as np
import torch
from torch.nn import Module, MSELoss
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm.auto import trange
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

def init_training_data(d: int, n0: int):
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

class TorchFMQA(Module, BaseEstimator, TransformerMixin):
    """FMQAの学習処理

    Args:
        Module: Base class for all neural network modules.
        BaseEstimator: Base class for all estimators in scikit-learn.
        TransformerMixin: Mixin class for all transformers in scikit-learn.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def set_grobal_params(self, params) -> None:
        self.params = params
        self.v   = self.get_grobal_params()["v"]
        self.w   = self.get_grobal_params()["w"]
        self.w0  = self.get_grobal_params()["w0"]

    def get_grobal_params(self) -> dict:
        return self.params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """入力 x を受け取って y の推定値を出力する

        Args:
            x (torch.Tensor): (データ数 × d) の 2 次元 tensor

        Returns:
            torch.Tensor: y の推定値 の 1次元 tensor (サイズはデータ数)
        """
        out_linear = torch.matmul(x, self.w) + self.w0

        out_1 = torch.matmul(x, self.v).pow(2).sum(1)
        out_2 = torch.matmul(x.pow(2), self.v.pow(2)).sum(1)
        out_quadratic = 0.5 * (out_1 - out_2)

        out = out_linear + out_quadratic
        return out

    def fit(self, X: np.ndarray, y: np.ndarray):
        # イテレーション数
        epochs = 2000

        # モデルの最適化関数 (パラメータ更新式)
        optimizer = torch.optim.AdamW([self.v, self.w, self.w0], lr=0.1)

        # 損失関数
        loss_func = MSELoss()

        # データセットの用意 (from numpy to tensor)
        x_tensor, y_tensor = (
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
        )

        dataset = TensorDataset(x_tensor, y_tensor) # (教師データ、正解データ)
        train_set, valid_set = random_split(dataset, [0.8, 0.2]) # 8:2に分割
        train_loader = DataLoader(train_set, batch_size=8, shuffle=True) # ミニバッチ学習、バッチサイズ: 8
        valid_loader = DataLoader(valid_set, batch_size=8, shuffle=True)

        # 学習の実行
        min_loss = 1e18  # 損失関数の最小値を保存
        best_state = self.state_dict()  # モデルの最も良いパラメータを保存 (今のモデルのパラメータ状態を保存)

        # `range` の代わりに `tqdm` モジュールを用いて進捗を表示
        for _ in trange(epochs, leave=False):
            # 学習フェイズ
            for x_train, y_train in train_loader:
                optimizer.zero_grad()
                pred_y = self.forward(x_train) # nnに代入 (forward)
                loss = loss_func(pred_y, y_train)
                loss.backward()
                optimizer.step()

            # 検証フェイズ
            with torch.no_grad():
                loss = 0
                # ミニバッチ
                for x_valid, y_valid in valid_loader:
                    out_valid = self.forward(x_valid)
                    loss += loss_func(out_valid, y_valid)
                if loss < min_loss:
                    # 損失関数の値が更新されたらパラメータを保存
                    best_state = copy.deepcopy(self.state_dict())
                    min_loss = loss

        # モデルを学習済みパラメータで更新
        self.load_state_dict(best_state)
        self.set_grobal_params({'v': self.v, 'w': self.w, 'w0': self.w0})

        return self

class AnnealFMQA(OptimizerMixin):
    """FMQAのアニーリング処理

    Args:
        OptimizerMixin: qudasの最適化用mixinクラス
    """

    def __init__(self, blackbox, d: int, token: str=None):
        self.blackbox   = blackbox
        self.d          = d
        self.token      = token
        self.result     = None

    def set_grobal_params(self, params) -> None:
        self.params = params

    def get_grobal_params(self) -> dict:
        return self.params

    def optimize(self, X=None, y=None) -> None:

        # 長さ d のバイナリ変数の配列を作成
        gen = VariableGenerator()
        x = gen.array("Binary", self.d)

        # TorchFM からパラメータ v, w, w0 を取得
        v   = self.get_grobal_params()["v"]
        w   = self.get_grobal_params()["w"]
        w0  = self.get_grobal_params()["w0"]

        # 目的関数を作成
        out_linear = w0 + (x * w).sum()
        out_1 = ((x[:, np.newaxis] * v).sum(axis=0) ** 2).sum()  # type: ignore
        out_2 = ((x[:, np.newaxis] * v) ** 2).sum()
        objective: Poly = out_linear + (out_1 - out_2) / 2
        objective = objective.sum()

        # 組合せ最適化モデルを構築
        amplify_model = Model(objective)

        # ソルバーの設定
        client = FixstarsClient()
        # ローカル環境等で実行する場合はコメントを外して Amplify AEのアクセストークンを入力してください
        client.token = self.token
        # 最適化の実行時間を 2 秒に設定
        client.parameters.timeout = timedelta(milliseconds=2000)

        # 最小化を実行
        result = solve(amplify_model, client)
        if len(result.solutions) == 0:
            raise RuntimeError("No solution was found.")

        # モデルを最小化する入力ベクトルを返却
        self.result = x.evaluate(result.best.values).astype(int)

        return self

class PipeIteration(IteratorMixin):
    """FMQAの繰り返し処理

    Args:
        IteratorMixin: qudasの繰り返し用mixinクラス
    """

    def __init__(self, blackbox, d: int, loop_num: int):
        self.blackbox   = blackbox
        self.d          = d
        self.loop_num   = loop_num
        self.models     = None
        self.results    = None

    def set_grobal_params(self, params) -> None:
        self.params = params

    def get_grobal_params(self) -> dict:
        return self.params

    def next_step(self, X, y=None, **iter_params) -> tuple:

        # self.results[1] が重複しないようにする
        while (self.results[1] == X).all(axis=1).any():
            flip_idx = rng.choice(np.arange(self.d))
            self.results[1][flip_idx] = 1 - self.results[1][flip_idx]

        # 推定された入力ベクトルを用いてブラックボックス関数を評価
        y_hat = self.blackbox(self.results[1])

        # 評価した値をデータセットに追加
        x = np.vstack((X, self.results[1]))
        y = np.append(y, y_hat)

        print(f"FMQA cycle: found y = {y_hat}; current best = {np.min(y)}")

        return x, y

if __name__ == '__main__':
    # 適当な関数を作成 (d次元, y = xQx)
    d = 100
    blackbox = make_blackbox_func(d)

    # 初期教師データの数
    N0 = 60
    x, y = init_training_data(d, N0)
    print(f"{x.shape=}, {y.shape=}")

    # FMQA サイクルの実行回数
    N = 10

    # 初期パラメータ
    k = 10
    v = torch.randn((d, k), requires_grad=True)
    w = torch.randn((d,), requires_grad=True)
    w0 = torch.randn((), requires_grad=True)
    parameters = {'v': v, 'w': w, 'w0': w0}

    # pipeline
    steps = [
        ('TorchFMQA', TorchFMQA()),
        ('AnnealFMQA', AnnealFMQA(blackbox, d, token="AE/p2lAwBrQpyGlHPKuMgpwbfQiO0OXXg6B")),
        ('pipeIteration', PipeIteration(blackbox, d, loop_num=N))
    ]

    pipe = Pipeline(steps)
    pipe.set_grobal_params(parameters)

    # 最適化
    result = pipe.optimize(x, y)
    print(f"{result=}")
    # print(f"{pipe.get_grobal_params()=}")
