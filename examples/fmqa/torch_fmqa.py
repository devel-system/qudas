# qudas & pipline
from sklearn.base import BaseEstimator, TransformerMixin

# module
import copy
import numpy as np
import torch
from torch.nn import Module, MSELoss
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm.auto import trange


class TorchFMQA(Module, BaseEstimator, TransformerMixin):
    """FMQAの学習処理

    Args:
        Module: Base class for all neural network modules.
        BaseEstimator: Base class for all estimators in scikit-learn.
        TransformerMixin: Mixin class for all transformers in scikit-learn.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.params = {}

    def set_global_params(self, params) -> None:
        """グローバルパラメータを設定"""
        self.params = params

    def get_global_params(self) -> dict:
        """グローバルパラメータを取得"""
        return self.params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """入力 x を受け取って y の推定値を出力する

        Args:
            x (torch.Tensor): (データ数 × d) の 2 次元 tensor

        Returns:
            torch.Tensor: y の推定値 の 1次元 tensor (サイズはデータ数)
        """

        v = self.params['v']
        w = self.params['w']
        w0 = self.params['w0']

        out_linear = torch.matmul(x, w) + w0

        out_1 = torch.matmul(x, v).pow(2).sum(1)
        out_2 = torch.matmul(x.pow(2), v.pow(2)).sum(1)
        out_quadratic = 0.5 * (out_1 - out_2)
        out = out_linear + out_quadratic
        return out

    def fit(self, X: np.ndarray, y: np.ndarray):
        """モデルの学習"""

        v = self.params['v']
        w = self.params['w']
        w0 = self.params['w0']

        # イテレーション数
        epochs = 2000

        # モデルの最適化関数 (パラメータ更新式)
        optimizer = torch.optim.AdamW([v, w, w0], lr=0.1)

        # 損失関数
        loss_func = MSELoss()

        # データセットの用意 (from numpy to tensor)
        x_tensor, y_tensor = (
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
        )

        dataset = TensorDataset(x_tensor, y_tensor)  # (教師データ、正解データ)
        train_set, valid_set = random_split(dataset, [0.8, 0.2])  # 8:2に分割
        train_loader = DataLoader(
            train_set, batch_size=8, shuffle=True
        )  # ミニバッチ学習、バッチサイズ: 8
        valid_loader = DataLoader(valid_set, batch_size=8, shuffle=True)

        # 学習の実行
        min_loss = 1e18  # 損失関数の最小値を保存
        best_state = (
            self.state_dict()
        )  # モデルの最も良いパラメータを保存 (今のモデルのパラメータ状態を保存)

        # `range` の代わりに `tqdm` モジュールを用いて進捗を表示
        for _ in trange(epochs, leave=False):
            # 学習フェイズ
            for x_train, y_train in train_loader:
                optimizer.zero_grad()
                pred_y = self.forward(x_train)  # nnに代入 (forward)
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

        # 次のパラメータを設定
        self.params['v'] = v
        self.params['w'] = w
        self.params['w0'] = w0
        self.set_global_params(self.params)

        return self
