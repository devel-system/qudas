from typing import Sequence

class Pipeline():
    def __init__(self, steps: Sequence[tuple]) -> None:
        """
        Pipelineクラスは一連のステップを受け取り、それぞれのステップを順に実行する。

        Args:
            steps (list): ステップのリスト。各ステップは (名前, オブジェクト) のタプル形式。
        """
        self.steps      = steps
        self.models = {step_name: None for step_name, _ in steps}
        self.results = {step_name: None for step_name, _ in steps}
        self.global_params = {}

    def set_grobal_params(self, params) -> None:
        """
        パイプライン全体に適用するグローバルパラメータを設定する。

        Args:
            params (dict): グローバルパラメータ。
        """
        for step in self.steps:
            if hasattr(step[1], 'set_global_params'):
                step[1].set_global_params(params)
        self.global_params = params

    def get_grobal_params(self) -> dict:
        """
        パイプライン全体に適用されたグローバルパラメータを取得する。

        Returns:
            dict: グローバルパラメータ。
        """
        return self.global_params

    def _split_steps(self):
        """
        ステップを中間ステップと最終ステップに分ける。
        """
        return self.steps[:-1], self.steps[-1]

    def _process_transform(self, step, X):
        """
        Transformerステップを実行する。

        Args:
            step: 現在のステップ。
            X: 入力データ。

        Returns:
            X: 変換されたデータ。
        """
        if hasattr(step[1], 'transform'):
            return step[1].transform(X)
        return X

    def _process_fit(self, step, X, y):
        """
        Fitステップを実行する。

        Args:
            step: 現在のステップ。
            X: 入力データ。
            y: ターゲットデータ。

        Returns:
            model: 訓練されたモデル。
        """
        if hasattr(step[1], 'fit'):
            return step[1].fit(X, y)
        return None

    def _process_optimize(self, step, X, y):
        """
        Optimizerステップを実行する。

        Args:
            step: 現在のステップ。
            X: 入力データ。
            y: ターゲットデータ。

        Returns:
            result: 最適化の結果。
        """
        if hasattr(step[1], 'optimize'):
            return step[1].optimize(X, y)
        return None

    def fit(self, X, y=None):
        """
        各ステップを順に適用してデータを訓練する。

        Args:
            X: 入力データ。
            y: ターゲットデータ。

        Returns:
            self: パイプラインオブジェクト自身。
        """

        middle_steps, last_step = self._split_steps()

        # middle_stepsで処理
        for _, step in enumerate(middle_steps):
            if hasattr(step[1], 'transform'):
                X = step[1].transform(X)
            if hasattr(step[1], 'optimize'):
                self.results[step[0]] = step[1].optimize(X, y).result
            if hasattr(step[1], 'fit'):
                self.models[step[0]] = step[1].fit(X, y)

        # 最後のステップのループ処理
        loop_num = getattr(last_step[1], 'loop_num', 1)

        while loop_num > 0:
            if hasattr(last_step[1], 'next_step'):
                last_step[1].models = self.models
                last_step[1].results = self.results
                X, y = last_step[1].next_step(X, y)
                loop_num -= 1
            else:
                if hasattr(last_step[1], 'transform'):
                    X = last_step[1].transform(X)
                if hasattr(last_step[1], 'optimize'):
                    self.results[last_step[0]] = last_step[1].optimize(X, y).result
                if hasattr(last_step[1], 'fit'):
                    self.models[last_step[0]] = last_step[1].fit(X, y)
                break

        return self

    def optimize(self, X=None, y=None):
        """
        各ステップを順に適用して最適化を実行する。

        Args:
            X: 入力データ。
            y: ターゲットデータ。

        Returns:
            self: パイプラインオブジェクト自身。
        """

        middle_steps, last_step = self._split_steps()

        # middle_stepsで処理
        for i, step in enumerate(middle_steps):
            # Transformer
            if hasattr(step[1], 'transform'):
                X = step[1].transform(X)

            # Optimizer
            if hasattr(step[1], 'optimize'):
                self.results[step[0]] = step[1].optimize(X, y).result
                self.set_global_params(step[1].get_global_params())

            # Estimator
            if hasattr(step[1], 'fit'):
                self.models[step[0]] = step[1].fit(X, y)
                self.set_global_params(step[1].get_global_params())

        # 最後のステップのループ処理
        loop_num = getattr(last_step[1], 'loop_num', 1)

        while loop_num > 0:
            if hasattr(last_step[1], 'next_step'):
                last_step[1].models = self.models
                last_step[1].results = self.results
                X, y = last_step[1].next_step(X, y)
                loop_num -= 1
            else:
                if hasattr(last_step[1], 'transform'):
                    X = last_step[1].transform(X)
                if hasattr(last_step[1], 'optimize'):
                    self.results[last_step[0]] = last_step[1].optimize(X, y).result
                if hasattr(last_step[1], 'fit'):
                    self.models[last_step[0]] = last_step[1].fit(X, y)
                break

        return self.results

    def predict(self, X):
        """
        各ステップを順に適用してデータを予測する。

        Args:
            X: 入力データ。

        Returns:
            X: 予測結果。
        """

        try:
            for step_name, step in self.steps:
                if hasattr(step[1], 'transform'):
                    X = self._process_transform(step, X)
                if hasattr(self.models[step_name], 'predict'):
                    return self.models[step_name].predict(X)
            raise RuntimeError("predictメソッドが見つかりませんでした。")
        except Exception as e:
            raise RuntimeError(f"予測中にエラーが発生しました: {str(e)}")

    def get_results(self):
        """
        最適化結果を取得する。

        Returns:
            dict: 各ステップごとの最適化結果。
        """
        return self.results

    def get_models(self):
        """
        学習されたモデルを取得する。

        Returns:
            dict: 各ステップごとのモデル。
        """
        return self.models