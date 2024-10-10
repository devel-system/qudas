# qudas & pipline
from qudas.pipeline.steps import OptimizerMixin

# module
from amplify import VariableGenerator, Model, FixstarsClient, solve, Poly
from datetime import timedelta
import numpy as np

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

    def set_global_params(self, params) -> None:
        self.params = params

    def get_global_params(self) -> dict:
        return self.params

    def optimize(self, X=None, y=None) -> None:

        # 長さ d のバイナリ変数の配列を作成
        gen = VariableGenerator()
        x = gen.array("Binary", self.d)

        # TorchFM からパラメータ v, w, w0 を取得
        v   = self.params["v"]
        w   = self.params["w"]
        w0  = self.params["w0"]

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