import unittest
from qudas.pipeline import Pipeline
from qudas.pipeline.steps.iterator_mixin import IteratorMixin
from qudas.pipeline.steps.optimizer_mixin import OptimizerMixin
from sklearn.base import BaseEstimator


# サンプルステップを定義
class SimpleEstimatorStep(BaseEstimator):
    """Sample Step for testing fit and predict"""

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def fit(self, X=None, y=None):
        """最適化処理"""
        self.params['alpha'] = self.params['alpha'] * 2  # パラメータの修正
        return self

    def predict(self, X=None, y=None):
        """最適化処理"""
        return self.params['alpha'] * X + 1  # alphe * X + 1

    def transform(self, X):
        """デフォルトで行う変換処理"""
        return X + 1  # 1を加算


class SimpleOptimizerStep(OptimizerMixin):
    """Sample Step for testing optimization"""

    def __init__(self):
        super().__init__()

    def optimize(self, X=None, y=None):
        """最適化処理"""
        return X * 2  # 倍増

    def transform(self, X):
        """デフォルトで行う変換処理"""
        return X + 1  # 1を加算


class SimpleIteratorStep(IteratorMixin):
    """Sample Step for testing iteration"""

    def __init__(self, loop_num: int):
        super().__init__(loop_num)

    def next_params(self, X, y=None):
        """次のパラメータに変更"""
        return X + 1, y


# Pipelineテストクラス
class TestPipeline(unittest.TestCase):

    def test_fit(self):
        """fitメソッドのテスト"""
        X = 10
        params = {'alpha': 0.1}
        steps = [
            (
                'SimpleEstimatorStep',
                SimpleEstimatorStep(params),
            ),  # Estimatorを使ったステップ
        ]
        pipeline = Pipeline(steps)
        pipeline.fit(X)

        self.assertEqual(
            pipeline.results['SimpleEstimatorStep'], None
        )  # サンプルでは predict しないためNone
        self.assertEqual(pipeline.models['SimpleEstimatorStep'].params, {'alpha': 0.2})

    def test_optimize(self):
        """optimizeメソッドのテスト"""

        steps = [
            (
                'SimpleOptimizerStep1',
                SimpleOptimizerStep(),
            ),  # OptimizerMixinを使ったステップ
            ('SimpleOptimizerStep2', SimpleOptimizerStep()),
        ]
        pipeline = Pipeline(steps)

        X = 10
        result = pipeline.optimize(X)
        self.assertEqual(result['SimpleOptimizerStep1'], 22)  # (X + 1) * 2 の結果を確認
        self.assertEqual(
            result['SimpleOptimizerStep2'], 24
        )  # {(X + 1) + 1} * 2 の結果を確認

    def test_predict(self):
        """predictメソッドのテスト"""
        X = 10
        params = {'alpha': 0.1}
        steps = [
            (
                'SimpleEstimatorStep',
                SimpleEstimatorStep(params),
            ),  # Estimatorを使ったステップ
        ]
        pipeline = Pipeline(steps)
        pipeline.fit(X)
        results = pipeline.predict(X)

        self.assertEqual(results['SimpleEstimatorStep'], 3.2)  # 0.2 * (X + 1) + 1

    def test_iterator(self):
        """イテレータステップのテスト"""
        steps = [
            (
                'SimpleOptimizerStep',
                SimpleOptimizerStep(),
            ),  # OptimizerMixinを使ったステップ
        ]
        pipeline = Pipeline(steps, iterator=SimpleIteratorStep(loop_num=2))

        X = 10
        result = pipeline.optimize(X)
        self.assertEqual(
            result['SimpleOptimizerStep'], 26
        )  # {(X + 1 + 1) + 1} * 2 の結果を確認

    def test_global_params(self):
        """グローバルパラメータのテスト"""
        params = {'alpha': 0.1}
        steps = [
            (
                'SimpleOptimizerStep',
                SimpleOptimizerStep(),
            ),  # OptimizerMixinを使ったステップ
        ]
        pipeline = Pipeline(steps, iterator=SimpleIteratorStep(loop_num=2))
        pipeline.set_global_params(params)

        self.assertEqual(pipeline.get_global_params(), params)


if __name__ == '__main__':
    unittest.main()
