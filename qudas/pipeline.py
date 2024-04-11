from typing import Sequence

class Pipeline():
    def __init__(self, steps: Sequence[tuple]) -> None:
        self.steps      = steps
        self.models     = [None for i in range(len(steps))]
        self.results    = [None for i in range(len(steps))]
        self.params     = None

    def set_grobal_params(self, params) -> None:
        for step in self.steps:
            step[1].set_grobal_params(params)

        self.params = params

    def get_grobal_params(self) -> dict:
        return self.params

    def _split_steps(self):
        return self.steps[:-1], self.steps[-1]

    def fit(self, X, y=None):
        """学習
        再度のステップがEstimator

        Args:
            X (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        middle_steps, last_step = self._split_steps()

        # Pipeline
        for i, step in enumerate(middle_steps):
            if hasattr(step[1], 'transform'):
                X = step[1].transform(X)

            # Optimizer
            if hasattr(step[1], 'optimize'):
                self.result[i] = step[1].optimize(X, y).result

            # Estimator
            if hasattr(step[1], 'fit'):
                self.models[i] = step[1].fit(X, y)

        # PipeIterator
        if hasattr(last_step[1], 'next_step'):

            # loop回数を取得
            if hasattr(self, 'loop_num'):
                self.loop_num -= 1
            elif isinstance(last_step[1].loop_num, int):
                self.loop_num = last_step[1].loop_num
            elif last_step[1].loop_num is None:
                raise ValueError('loop_numは必須です。')
            else:
                raise TypeError('loop_numはint型です。')

            # 終了判定
            if self.loop_num == 0:
                return self

            else:
                # 学習モデルと結果を共有
                last_step[1].models = self.models
                last_step[1].results = self.results

                # 次のステップを実行
                X, y = last_step[1].next_step(X, y)
                return self.fit(X, y)

        return self

    def optimize(self, X=None, y=None):
        _, last_step = self._split_steps()

        # Pipeline
        for i, step in enumerate(self.steps):
            if hasattr(step[1], 'transform'):
                X = step[1].transform(X)

            # Optimizer
            if hasattr(step[1], 'optimize'):
                self.results[i] = step[1].optimize(X, y).result
                self.set_grobal_params(step[1].get_grobal_params())

            # Estimator
            if hasattr(step[1], 'fit'):
                self.models[i] = step[1].fit(X, y)
                self.set_grobal_params(step[1].get_grobal_params())

        # PipeIterator
        if hasattr(last_step[1], 'next_step'):

            # loop回数を取得
            if hasattr(self, 'loop_num'):
                self.loop_num -= 1
                last_step[1].loop_num -= 1
            elif isinstance(last_step[1].loop_num, int):
                self.loop_num = last_step[1].loop_num
            elif last_step[1].loop_num is None:
                raise ValueError('loop_numは必須です。')
            else:
                raise TypeError('loop_numはint型です。')

            # 終了判定
            if self.loop_num == 0:
                return self.results[-2]

            else:
                # 学習モデルと結果を共有
                last_step[1].models = self.models
                last_step[1].results = self.results

                # 次のステップを実行
                X, y = last_step[1].next_step(X, y)
                return self.optimize(X, y)

        return self.results[-1]