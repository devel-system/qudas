# qudas
from qudas.base import OptimizerMixin
from qudas.pipeline import Pipeline

from amplify import VariableGenerator, Model, FixstarsClient, solve
from datetime import timedelta

class TestAnnealing(OptimizerMixin):
    def __init__(self, token: str=None):
        self.token  = token
        self.result = None

    def optimize(self, X=None, y=None) -> None:

        gen = VariableGenerator()
        q = gen.array("Binary", shape=(3))
        objective = q[0] * q[1] * q[2]

        # ソルバーの設定
        client = FixstarsClient()
        # ローカル環境等で実行する場合はコメントを外して Amplify AEのアクセストークンを入力してください
        client.token = self.token
        # 最適化の実行時間を 2 秒に設定
        client.parameters.timeout = timedelta(milliseconds=100)

        # 最小化を実行
        result = solve(Model(objective), client)
        if len(result.solutions) == 0:
            raise RuntimeError("No solution was found.")

        # 最小値を返却
        self.result = { "q": q.evaluate(result.best.values), "objective": result.best.objective }

        return self

if __name__ == '__main__':

    # pipeline
    steps = [
        ('testAnnealing', TestAnnealing(token="AE/p2lAwBrQpyGlHPKuMgpwbfQiO0OXXg6B"))
    ]

    pipe = Pipeline(steps)
    result = pipe.optimize()

    print(result)