from typing import Sequence, Dict, Any, Tuple, Optional
from .steps import IteratorMixin

class Pipeline:
    def __init__(self, steps: Sequence[Tuple[str, Any]], iterator: Optional[IteratorMixin]=None) -> None:
        """
        Pipelineクラスは一連のステップを受け取り、それぞれのステップを順に実行する。

        Args:
            steps (Sequence[Tuple[str, Any]]): ステップのリスト。各ステップは (名前, オブジェクト) のタプル形式。
            iterator (Optional[IteratorMixin]): Pipeline全体を繰り返すイテレータ。イテレータは IteratorMixin 形式。デフォルト値はNone。
        """
        self.steps = steps
        self.models = {step_name: None for step_name, _ in steps}
        self.results = {step_name: None for step_name, _ in steps}
        self.global_params = {}
        self.global_iterator = iterator

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
            mode (str): 実行モード ('fit', 'transform', 'optimize', 'predict')。

        Returns:
            Any: ステップまたはモデルによって処理されたデータ。
        """
        step_name, step_instance = step

        # ステップに対応するモデルがある場合、そのモデルのtransformやpredictを使用
        model = self.models[step_name]
        if model is not None:
            if mode == 'transform' and hasattr(model, 'transform'):
                return model.transform(X)
            if mode == 'predict' and hasattr(model, 'predict'):
                return model.predict(X)

        # モデルがない場合、通常のステップ処理
        if mode == 'transform':
            if hasattr(step_instance, 'transform'):
                return step_instance.transform(X)
            else:
                return X

        elif mode == 'fit' and hasattr(step_instance, 'fit'):
            return step_instance.fit(X, y)

        elif mode == 'optimize' and hasattr(step_instance, 'optimize'):
            return step_instance.optimize(X, y)

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

        # 全体のglobal_iteratorのループ回数を取得
        global_loop_num = getattr(self.global_iterator, 'loop_num', 1)

        while global_loop_num > 0:
            # middle_stepsと最後のステップを分離
            middle_steps, last_step = self._split_steps()

            # middle_stepsで処理
            for step in middle_steps:

                step_name, step_instance = step

                # 各ステップごとのループ回数を取得 (IteratorMixinのloop_num)
                step_loop_num = getattr(step_instance, 'loop_num', 1)

                while step_loop_num > 0:
                    X = self._process_step(step, X, y, 'transform')

                    # optimize を実行し、結果を y に格納
                    if hasattr(step_instance, 'optimize'):
                        y = self._process_step(step, X, y, 'optimize')

                    # fit を実行し、モデルを保存
                    if hasattr(step_instance, 'fit'):
                        self.models[step_name] = self._process_step(step, X, y, 'fit')

                    # next_step が定義されていれば、次のパラメータを取得
                    if hasattr(step_instance, 'next_params'):
                        X, y = step_instance.next_params(X, y)

                    # ステップごとのループ回数をデクリメント
                    step_loop_num -= 1

            # 最後のステップの処理
            X = self._process_step(last_step, X, y, 'transform')

            # 最後のステップで optimize と fit を実行し、optimize の結果を y に格納
            if hasattr(last_step[1], 'optimize'):
                y = self._process_step(last_step, X, y, 'optimize')

            if hasattr(last_step[1], 'fit'):
                self.models[last_step[0]] = self._process_step(last_step, X, y, 'fit')

            # next_step が定義されていれば、次のパラメータを取得
            if hasattr(last_step[1], 'next_params'):
                X, y = last_step[1].next_params(X, y)

            # 全体のループ回数をデクリメント
            global_loop_num -= 1

        return self

    def optimize(self, X: Any = None, y: Any = None) -> Dict[str, Any]:
        """
        各ステップを順に適用して最適化を実行する。

        Args:
            X (Any): 入力データ。
            y (Any): ターゲットデータ。

        Returns:
            Dict[str, Any]: 各ステップの最適化結果。

        Raises:
            RuntimeError: 最後のステップに optimize メソッドがない場合。
        """

        # 全体のglobal_iteratorのループ回数を取得
        global_loop_num = getattr(self.global_iterator, 'loop_num', 1)

        while global_loop_num > 0:

            # self.stepsで処理
            for step in self.steps:
                step_name, step_instance = step

                # 各ステップのループ回数を取得 (IteratorMixinのloop_num)
                step_loop_num = getattr(step_instance, 'loop_num', 1)

                while step_loop_num > 0:

                    # Transformer
                    X = self._process_step(step, X, y, 'transform')

                    # Estimator
                    if hasattr(step_instance, 'fit'):

                        # パラメータをstepと共有（処理前）
                        step_instance.set_global_params(self.get_global_params())
                        step_instance.models = self.models
                        step_instance.results = self.results

                        self.models[step_name] = self._process_step(step, X, y, 'fit')

                        # パラメータをstepと共有（処理後）
                        step_instance.models = self.models
                        self.set_global_params(step_instance.get_global_params())

                    if global_loop_num == 1 and step_loop_num == 1 and step_name == self.steps[-1][0]:

                        # Optimizer
                        if hasattr(step_instance, 'optimize'):

                            # パラメータをstepと共有（処理前）
                            step_instance.set_global_params(self.get_global_params())
                            step_instance.models = self.models
                            step_instance.results = self.results

                            self.results[step_name] = self._process_step(step, X, y, 'optimize')

                            # パラメータをstepと共有（処理後）
                            step_instance.results = self.results
                            self.set_global_params(step_instance.get_global_params())

                            return self.results

                        else:
                            # optimize メソッドが見つからなかった場合のエラー
                            raise RuntimeError("パイプラインの最後のステップに optimize メソッドが見つかりませんでした。")

                    else:
                        # Optimizer
                        if hasattr(step_instance, 'optimize'):

                            # パラメータをstepと共有（処理前）
                            step_instance.set_global_params(self.get_global_params())
                            step_instance.models = self.models
                            step_instance.results = self.results

                            self.results[step_name] = self._process_step(step, X, y, 'optimize')

                            # パラメータをstepと共有（処理後）
                            step_instance.results = self.results
                            self.set_global_params(step_instance.get_global_params())

                    # next_params が定義されていれば、次のパラメータを取得
                    if hasattr(step_instance, 'next_params'):

                        # パラメータをstepと共有
                        step_instance.set_global_params(self.get_global_params())
                        step_instance.models = self.models
                        step_instance.results = self.results

                        X, y = step_instance.next_params(X, y)
                        self.set_global_params(step_instance.get_global_params())

                    # ステップごとのループ回数をデクリメント
                    step_loop_num -= 1

            # next_params が定義されていれば、次のパラメータを取得
            if hasattr(self.global_iterator, 'next_params'):

                # パラメータをstepと共有
                self.global_iterator.set_global_params(self.get_global_params())
                self.global_iterator.models = self.models
                self.global_iterator.results = self.results

                X, y = self.global_iterator.next_params(X, y)
                self.set_global_params(self.global_iterator.get_global_params())

            # ステップごとのループ回数をデクリメント
            global_loop_num -= 1

    def predict(self, X: Any) -> Any:
        """
        各ステップを順に適用してデータを予測する。予測を行うためには、最後のステップで
        IteratorMixin でないステップが predict メソッドを持っている必要がある。

        Args:
            X (Any): 入力データ。

        Returns:
            Any: 予測結果。

        Raises:
            RuntimeError: 最後のステップに predict メソッドがない場合。
        """

        middle_steps, last_step = self._split_steps()

        # 中間ステップのyを初期化
        y = None

        # 最後のステップのループ回数を取得 (IteratorMixinのloop_num。IteratorMixinがない場合は1)
        loop_num = getattr(last_step[1], 'loop_num', 1)

        # 全ステップを loop_num 回繰り返す
        while loop_num > 0:
            # middle_stepsで処理
            for step in middle_steps:
                model = self.models[step[0]]

                # transform 処理
                X = self._process_step(step, X, y, 'transform')

                # モデルが存在する場合の predict 処理
                if model is not None and hasattr(model, 'predict'):
                    y = model.predict(X)

                # optimize がある場合の処理
                elif hasattr(step[1], 'optimize'):
                    y = self._process_step(step, X, y, 'optimize')

            # 最後のステップの transform 処理
            X = self._process_step(last_step, X, y, 'transform')

            # 最後のステップに predict メソッドがある場合、その処理を実行
            if hasattr(last_step[1], 'predict'):
                y = self._process_step(last_step, X, y, 'predict')

            # 最後のステップに next_params がある場合、次のパラメータを取得
            if hasattr(last_step[1], 'next_params'):
                X, y = last_step[1].next_params(X, y)

            # ループ回数をデクリメント
            loop_num -= 1

        # 最後のステップで predict を実行
        final_predict_step = last_step
        if isinstance(last_step[1], IteratorMixin):
            # IteratorMixinを持つ場合は、その前のステップで predict を実行
            final_predict_step = middle_steps[-1] if middle_steps else last_step

        # 最終的な predict の実行
        if self.models[final_predict_step[0]] and hasattr(self.models[final_predict_step[0]], 'predict'):
            return self._process_step(final_predict_step, X, y, 'predict')

        # predict メソッドが見つからなかった場合のエラー
        raise RuntimeError("パイプラインの最後のステップに predict メソッドが見つかりませんでした。")

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
