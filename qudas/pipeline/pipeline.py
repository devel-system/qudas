# qudas/pipeline/pipeline.py
from __future__ import annotations

from typing import Sequence, Dict, Any, Tuple, Optional, Iterable, List, MutableMapping

from .artifacts import ArtifactBase
from .blocks.base_block import BaseBlock
from .converter import ArtifactConverterRegistry
from .base import QdBaseEstimator, QdBaseStep
from .steps import IteratorMixin


StepTuple = Tuple[str, QdBaseStep]


class QdPipeline(QdBaseEstimator):
    """
    scikit-learn 風の Pipeline 実装。

    - steps: List[Tuple[str, QdBaseStep]]
    - named_steps: dict アクセス
    - set_params(step__param=value) でステップのハイパーパラメータを一括設定
    - fit / transform / fit_transform / predict / optimize を提供
    - 実行時コンテキスト (context) は QdBaseEstimator の set_context で注入
    """

    def __init__(
        self,
        steps: Iterable[StepTuple],
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        steps:
            (name, step) のタプル列。name はユニークな識別子。
        memory:
            将来用。joblib.Memory 等のキャッシュを想定（現状未使用）。
        kwargs:
            QdBaseEstimator の追加引数（必要なら）。
            旧 API 互換のため ``iterator=...`` は ``global_iterator`` として扱う。
        """
        if "iterator" in kwargs and "global_iterator" not in kwargs:
            kwargs["global_iterator"] = kwargs.pop("iterator")
        super().__init__(**kwargs)
        self.steps: List[StepTuple] = self._validate_steps(list(steps))
        self.named_steps: MutableMapping[str, QdBaseStep] = {n: s for n, s in self.steps}

        if not hasattr(self, "models") or getattr(self, "models", None) is None:
            self.models = {}
        if not hasattr(self, "results") or getattr(self, "results", None) is None:
            self.results = {}
        if not hasattr(self, "global_params"):
            self.global_params = {}

        class _DefaultGlobalIterator:
            loop_num = 1

        if not hasattr(self, "global_iterator"):
            self.global_iterator = _DefaultGlobalIterator()

    def set_global_params(self, params: Dict[str, Any]) -> None:
        """
        パイプライン全体に適用するグローバルパラメータを設定する。

        Args:
            params (Dict[str, Any]): グローバルパラメータ。
        """
        self.global_params = params

    def get_global_params(self) -> Dict[str, Any]:
        """
        パイプライン全体に適用されたグローバルパラメータを取得する。

        Returns:
            Dict[str, Any]: グローバルパラメータ。
        """
        return self.global_params

    def _assign_global_params(self, step_instance):
        """ステップが global_params を自動的に持つようにする"""
        if not hasattr(step_instance, 'set_global_params'):
            step_instance.global_params = self.global_params
        else:
            step_instance.set_global_params(self.global_params)

        # models と results を共有
        step_instance.models = self.models
        step_instance.results = self.results

    def _update_params(self, step_instance):
        """パラメータを更新"""
        if hasattr(step_instance, 'get_global_params'):
            self.global_params = step_instance.get_global_params()

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

        # ============================================================
        # Pipeline Block（QuantumBlock / ClassicalBlock）
        # ============================================================
        if isinstance(step_instance, BaseBlock):
            # 入力が Artifact でない場合はそのままエラーにする
            if not isinstance(X, ArtifactBase):
                raise TypeError(
                    f"Block '{step_name}' expects ArtifactBase input, "
                    f"but got {type(X).__name__}"
                )

            expected_type = step_instance.expected_input_type

            # Artifact の自動変換
            X_converted = ArtifactConverterRegistry.convert(X, expected_type)

            # Block 実行（artifact -> artifact）
            return step_instance.run(X_converted)


        # ============================================================
        # 既存の pipeline の挙動
        # ============================================================
        model = self.models.get(step_name)

        # モデルが存在し、transformまたはpredictを実行可能な場合
        if model:
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

    def _validate_steps(self, steps: List[tuple]) -> List[tuple]:
        """
        steps の最小バリデーション。
        - (name, step) 形式
        - name は str でユニーク
        - step は None でない
        - step は fit/transform/predict/optimize/run のいずれかを持つ
        """
        if not isinstance(steps, list):
            raise TypeError("steps must be a list of (name, step) tuples.")

        names = []
        validated = []

        for i, item in enumerate(steps):
            if not (isinstance(item, tuple) and len(item) == 2):
                raise TypeError(f"steps[{i}] must be a tuple (name, step).")

            name, step = item

            if not isinstance(name, str) or not name:
                raise TypeError(f"steps[{i}].name must be a non-empty str.")
            if step is None:
                raise TypeError(f"steps[{i}].step must not be None.")

            # 実行可能性の最低限チェック（Step/Block 両対応）
            if not any(hasattr(step, m) for m in ("fit", "transform", "predict", "optimize", "run")):
                raise TypeError(
                    f"steps[{i}] '{name}' must implement at least one of "
                    f"fit/transform/predict/optimize/run."
                )

            names.append(name)
            validated.append((name, step))

        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"step names must be unique. duplicated: {sorted(dup)}")

        return validated

    def fit(self, X: Any, y: Any = None) -> 'Pipeline':
        """
        各ステップを順に適用してデータを訓練する。

        Args:
            X (Any): 入力データ。
            y (Any): ターゲットデータ。

        Returns:
            Pipeline: パイプラインオブジェクト自身。

        Raises:
            RuntimeError: 最後のステップに fit メソッドがない場合。
        """

        # 全体のglobal_iteratorのループ回数を取得
        global_loop_num = getattr(self.global_iterator, 'loop_num', 1)

        while global_loop_num > 0:

            # self.stepsで処理
            for step in self.steps:
                step_name, step_instance = step

                # 各ステップごとのループ回数を取得 (IteratorMixinのloop_num)
                step_loop_num = getattr(step_instance, 'loop_num', 1)

                while step_loop_num > 0:
                    X = self._process_step(step, X, y, 'transform')

                    # optimize を実行し、結果を y に格納
                    if hasattr(step_instance, 'optimize'):

                        # パラメータをstepと共有（処理前）
                        self._assign_global_params(step_instance)

                        self.results[step_name] = self._process_step(
                            step, X, y, 'optimize'
                        )

                        # パラメータをstepと共有（処理後）
                        step_instance.results = self.results
                        self._update_params(step_instance)

                    else:
                        # optimize を持たないステップ（例: 素の Estimator）では結果スロットを明示的に None
                        self.results[step_name] = None
                        step_instance.results = self.results

                    if (
                        global_loop_num == 1
                        and step_loop_num == 1
                        and step_name == self.steps[-1][0]
                    ):

                        # fit を実行し、モデルを保存
                        if hasattr(step_instance, 'fit'):

                            # パラメータをstepと共有（処理前）
                            self._assign_global_params(step_instance)

                            self.models[step_name] = self._process_step(
                                step, X, y, 'fit'
                            )

                            # パラメータをstepと共有（処理後）
                            step_instance.models = self.models
                            self._update_params(step_instance)

                        else:
                            # optimize メソッドが見つからなかった場合のエラー
                            raise RuntimeError(
                                "パイプラインの最後のステップに fit メソッドが見つかりませんでした。"
                            )

                    else:
                        # fit を実行し、モデルを保存
                        if hasattr(step_instance, 'fit'):

                            # パラメータをstepと共有（処理前）
                            self._assign_global_params(step_instance)

                            self.models[step_name] = self._process_step(
                                step, X, y, 'fit'
                            )

                            # パラメータをstepと共有（処理後）
                            step_instance.models = self.models
                            self._update_params(step_instance)

                    # next_params が定義されていれば、次のパラメータを取得
                    if hasattr(step_instance, 'next_params'):

                        # パラメータをstepと共有
                        self._assign_global_params(step_instance)

                        X, y = step_instance.next_params(X, y)
                        self._update_params(step_instance)

                    # ステップごとのループ回数をデクリメント
                    step_loop_num -= 1

            # next_params が定義されていれば、次のパラメータを取得
            if hasattr(self.global_iterator, 'next_params'):

                # パラメータをstepと共有
                self._assign_global_params(self.global_iterator)

                X, y = self.global_iterator.next_params(X, y)
                self._update_params(self.global_iterator)

            # ステップごとのループ回数をデクリメント
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
                        self._assign_global_params(step_instance)

                        self.models[step_name] = self._process_step(step, X, y, 'fit')

                        # パラメータをstepと共有（処理後）
                        step_instance.models = self.models
                        self._update_params(step_instance)

                    if (
                        global_loop_num == 1
                        and step_loop_num == 1
                        and step_name == self.steps[-1][0]
                    ):

                        # Optimizer
                        if hasattr(step_instance, 'optimize'):

                            # パラメータをstepと共有（処理前）
                            self._assign_global_params(step_instance)

                            self.results[step_name] = self._process_step(
                                step, X, y, 'optimize'
                            )

                            # パラメータをstepと共有（処理後）
                            step_instance.results = self.results
                            self._update_params(step_instance)

                            return self.results

                        else:
                            # optimize メソッドが見つからなかった場合のエラー
                            raise RuntimeError(
                                "パイプラインの最後のステップに optimize メソッドが見つかりませんでした。"
                            )

                    else:
                        # Optimizer
                        if hasattr(step_instance, 'optimize'):

                            # パラメータをstepと共有（処理前）
                            self._assign_global_params(step_instance)

                            self.results[step_name] = self._process_step(
                                step, X, y, 'optimize'
                            )

                            # パラメータをstepと共有（処理後）
                            step_instance.results = self.results
                            self._update_params(step_instance)

                    # next_params が定義されていれば、次のパラメータを取得
                    if hasattr(step_instance, 'next_params'):

                        # パラメータをstepと共有
                        self._assign_global_params(step_instance)

                        X, y = step_instance.next_params(X, y)
                        self._update_params(step_instance)

                    # ステップごとのループ回数をデクリメント
                    step_loop_num -= 1

            # next_params が定義されていれば、次のパラメータを取得
            if hasattr(self.global_iterator, 'next_params'):

                # パラメータをstepと共有
                self._assign_global_params(self.global_iterator)

                X, y = self.global_iterator.next_params(X, y)
                self._update_params(self.global_iterator)

            # ステップごとのループ回数をデクリメント
            global_loop_num -= 1

    def predict(self, X: Any = None) -> Dict[str, Any]:
        """
        各ステップを順に適用してデータを予測する。予測を行うためには、最後のステップで
        IteratorMixin でないステップが predict メソッドを持っている必要がある。

        Args:
            X (Any): 入力データ。

        Returns:
            Any: 予測結果。

        Raises:
            RuntimeError: 最後のステップに predict メソッドがない場合。最後のステップで model が作成されていない場合。
        """

        # y の初期値
        y = None

        # 全体のglobal_iteratorのループ回数を取得
        global_loop_num = getattr(self.global_iterator, 'loop_num', 1)

        while global_loop_num > 0:

            # self.stepsで処理
            for step in self.steps:
                step_name, step_instance = step

                # 各ステップのループ回数を取得 (IteratorMixinのloop_num)
                step_loop_num = getattr(step_instance, 'loop_num', 1)

                # 全ステップを step_loop_num 回繰り返す
                while step_loop_num > 0:

                    # Transformer
                    X = self._process_step(step, X, y, 'transform')

                    if (
                        global_loop_num == 1
                        and step_loop_num == 1
                        and step_name == self.steps[-1][0]
                    ):

                        # Predict
                        if hasattr(step_instance, 'predict'):

                            # パラメータをstepと共有（処理前）
                            self._assign_global_params(step_instance)

                            # predict 処理
                            model = self.models[step_name]
                            if model is not None:
                                self.results[step_name] = model.predict(X)

                                # パラメータをstepと共有（処理後）
                                step_instance.results = self.results
                                self._update_params(step_instance)

                                return self.results

                            else:
                                # model が見つからなかった場合のエラー
                                raise RuntimeError(
                                    "パイプラインの最後のステップに model が見つかりませんでした。"
                                )

                        else:
                            # predict メソッドが見つからなかった場合のエラー
                            raise RuntimeError(
                                "パイプラインの最後のステップに predict メソッドが見つかりませんでした。"
                            )

                    else:
                        # Predict
                        if hasattr(step_instance, 'predict'):

                            # パラメータをstepと共有（処理前）
                            self._assign_global_params(step_instance)

                            # predict 処理
                            model = self.models[step_name]
                            if model is not None:
                                self.results[step_name] = model.predict(X)

                                # パラメータをstepと共有（処理後）
                                step_instance.results = self.results
                                self._update_params(step_instance)

                    # optimize を実行し、結果を y に格納
                    if hasattr(step_instance, 'optimize'):

                        # パラメータをstepと共有（処理前）
                        self._assign_global_params(step_instance)

                        self.results[step_name] = self._process_step(
                            step, X, y, 'optimize'
                        )

                        # パラメータをstepと共有（処理後）
                        step_instance.results = self.results
                        self._update_params(step_instance)

                    # next_params が定義されていれば、次のパラメータを取得
                    if hasattr(step_instance, 'next_params'):

                        # パラメータをstepと共有
                        self._assign_global_params(step_instance)

                        X, y = step_instance.next_params(X, y)
                        self._update_params(step_instance)

                    # ステップごとのループ回数をデクリメント
                    step_loop_num -= 1

            # next_params が定義されていれば、次のパラメータを取得
            if hasattr(self.global_iterator, 'next_params'):

                # パラメータをstepと共有
                self._assign_global_params(self.global_iterator)

                X, y = self.global_iterator.next_params(X, y)
                self._update_params(self.global_iterator)

            # ステップごとのループ回数をデクリメント
            global_loop_num -= 1

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
