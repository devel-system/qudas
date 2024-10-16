from typing import Sequence, Dict, Any, Tuple


class Pipeline:
    def __init__(self, steps: Sequence[Tuple[str, Any]]) -> None:
        """
        Pipelineクラスは一連のステップを受け取り、それぞれのステップを順に実行する。

        Args:
            steps (Sequence[Tuple[str, Any]]): ステップのリスト。各ステップは (名前, オブジェクト) のタプル形式。
        """
        self.steps = steps
        self.models = {step_name: None for step_name, _ in steps}
        self.results = {step_name: None for step_name, _ in steps}
        self.global_params = {}

    def set_global_params(self, params: Dict[str, Any]) -> None:
        """
        パイプライン全体に適用するグローバルパラメータを設定する。

        Args:
            params (Dict[str, Any]): グローバルパラメータ。
        """
        for step in self.steps:
            if hasattr(step[1], 'set_global_params'):
                step[1].set_global_params(params)
        self.global_params = params

    def get_global_params(self) -> Dict[str, Any]:
        """
        パイプライン全体に適用されたグローバルパラメータを取得する。

        Returns:
            Dict[str, Any]: グローバルパラメータ。
        """
        return self.global_params

    def _split_steps(self) -> Tuple[Sequence[Tuple[str, Any]], Tuple[str, Any]]:
        """
        ステップを中間ステップと最終ステップに分ける。

        Returns:
            Tuple[Sequence[Tuple[str, Any]], Tuple[str, Any]]: 中間ステップと最終ステップのタプル。
        """
        return self.steps[:-1], self.steps[-1]

    def _process_step(self, step: Tuple[str, Any], X: Any, y: Any, mode: str) -> Any:
        """
        ステップを実行する。

        Args:
            step (Tuple[str, Any]): 現在のステップ。
            X (Any): 入力データ。
            y (Any): ターゲットデータ。
            mode (str): 実行モード ('fit', 'transform', 'optimize')。

        Returns:
            Any: ステップによって処理されたデータまたは結果。
        """
        if mode == 'transform':
            if hasattr(step[1], 'transform'):
                return step[1].transform(X)
            else:
                return X  # メソッドがない場合はXをそのまま返す
        elif mode == 'fit' and hasattr(step[1], 'fit'):
            return step[1].fit(X, y)
        elif mode == 'optimize' and hasattr(step[1], 'optimize'):
            return step[1].optimize(X, y).result
        return None

    def fit(self, X: Any, y: Any = None) -> 'Pipeline':
        """
        各ステップを順に適用してデータを訓練する。

        Args:
            X (Any): 入力データ。
            y (Any): ターゲットデータ。

        Returns:
            Pipeline: パイプラインオブジェクト自身。
        """

        middle_steps, last_step = self._split_steps()

        # middle_stepsで処理
        for step in middle_steps:
            self._process_step(step, X, y, 'transform')
            self.results[step[0]] = self._process_step(step, X, y, 'optimize')
            self.models[step[0]] = self._process_step(step, X, y, 'fit')

        # 最後のステップのループ処理
        loop_num = getattr(last_step[1], 'loop_num', 1)

        while loop_num > 0:
            if hasattr(last_step[1], 'next_step'):
                last_step[1].models = self.models
                last_step[1].results = self.results
                X, y = last_step[1].next_step(X, y)
                loop_num -= 1
            else:
                self._process_step(last_step, X, y, 'transform')
                self.results[last_step[0]] = self._process_step(
                    last_step, X, y, 'optimize'
                )
                self.models[last_step[0]] = self._process_step(last_step, X, y, 'fit')
                break

        return self

    def optimize(self, X: Any = None, y: Any = None) -> Dict[str, Any]:
        """
        各ステップを順に適用して最適化を実行する。

        Args:
            X (Any): 入力データ。
            y (Any): ターゲットデータ。

        Returns:
            Dict[str, Any]: 各ステップの最適化結果。
        """

        middle_steps, last_step = self._split_steps()

        # middle_stepsで処理
        for step in middle_steps:
            # Transformer
            X = self._process_step(step, X, y, 'transform')

            # Optimizer
            if hasattr(step[1], 'optimize'):
                self.results[step[0]] = self._process_step(step, X, y, 'optimize')
                self.set_global_params(step[1].get_global_params())

            # Estimator
            if hasattr(step[1], 'fit'):
                self.models[step[0]] = self._process_step(step, X, y, 'fit')
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
                self._process_step(last_step, X, y, 'transform')
                self.results[last_step[0]] = self._process_step(
                    last_step, X, y, 'optimize'
                )
                self.models[last_step[0]] = self._process_step(last_step, X, y, 'fit')
                break

        return self.results

    def predict(self, X: Any) -> Any:
        """
        各ステップを順に適用してデータを予測する。

        Args:
            X (Any): 入力データ。

        Returns:
            Any: 予測結果。
        """

        try:
            for step_name, step in self.steps:
                X = self._process_step(step, X, None, 'transform')
                if hasattr(self.models[step_name], 'predict'):
                    return self.models[step_name].predict(X)
            raise RuntimeError("predictメソッドが見つかりませんでした。")
        except Exception as e:
            raise RuntimeError(f"予測中にエラーが発生しました: {str(e)}")

    def get_results(self) -> Dict[str, Any]:
        """
        最適化結果を取得する。

        Returns:
            Dict[str, Any]: 各ステップごとの最適化結果。
        """
        return self.results

    def get_models(self) -> Dict[str, Any]:
        """
        学習されたモデルを取得する。

        Returns:
            Dict[str, Any]: 各ステップごとのモデル。
        """
        return self.models
