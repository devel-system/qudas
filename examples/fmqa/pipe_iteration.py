# qudas & pipline
from qudas.pipeline.steps import IteratorMixin

# module
import numpy as np

# 乱数シードの固定
seed = 1234
rng = np.random.default_rng(seed)

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

    def set_global_params(self, params) -> None:
        self.params = params

    def get_global_params(self) -> dict:
        return self.params

    def next_params(self, X, y=None, **iter_params) -> tuple:

        # self.results["AnnealFMQA"] が重複しないようにする
        while (self.results["AnnealFMQA"] == X).all(axis=1).any():
            flip_idx = rng.choice(np.arange(self.d))
            self.results["AnnealFMQA"][flip_idx] = 1 - self.results["AnnealFMQA"][flip_idx]

        # 推定された入力ベクトルを用いてブラックボックス関数を評価
        y_hat = self.blackbox(self.results["AnnealFMQA"])

        # 評価した値をデータセットに追加
        x = np.vstack((X, self.results["AnnealFMQA"]))
        y = np.append(y, y_hat)

        print(f"FMQA cycle: found y = {y_hat}; current best = {np.min(y)}")

        return x, y