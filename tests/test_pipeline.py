import unittest

from sklearn.base import BaseEstimator

from qudas.pipeline import QdPipeline
from qudas.pipeline.steps.iterator_mixin import IteratorMixin
from qudas.pipeline.steps.optimizer_mixin import OptimizerMixin


class SimpleEstimatorStep(BaseEstimator):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def fit(self, X=None, y=None):
        self.params['alpha'] = self.params['alpha'] * 2
        return self

    def predict(self, X=None, y=None):
        return self.params['alpha'] * X + 1

    def transform(self, X):
        return X + 1


class SimpleOptimizerStep(OptimizerMixin):
    def __init__(self):
        super().__init__()

    def optimize(self, X=None, y=None):
        return X * 2

    def transform(self, X):
        return X + 1


class SimpleIteratorStep(IteratorMixin):
    def __init__(self, loop_num: int):
        super().__init__(loop_num)

    def next_params(self, X, y=None):
        return X + 1, y


class TestPipeline(unittest.TestCase):
    def test_fit(self):
        X = 10
        params = {'alpha': 0.1}
        steps = [('SimpleEstimatorStep', SimpleEstimatorStep(params))]
        pipeline = QdPipeline(steps)
        pipeline.fit(X)
        self.assertEqual(pipeline.results['SimpleEstimatorStep'], None)
        self.assertEqual(pipeline.models['SimpleEstimatorStep'].params, {'alpha': 0.2})

    def test_optimize(self):
        steps = [
            ('SimpleOptimizerStep1', SimpleOptimizerStep()),
            ('SimpleOptimizerStep2', SimpleOptimizerStep()),
        ]
        pipeline = QdPipeline(steps)
        X = 10
        result = pipeline.optimize(X)
        self.assertEqual(result['SimpleOptimizerStep1'], 22)
        self.assertEqual(result['SimpleOptimizerStep2'], 24)

    def test_predict(self):
        X = 10
        params = {'alpha': 0.1}
        steps = [('SimpleEstimatorStep', SimpleEstimatorStep(params))]
        pipeline = QdPipeline(steps)
        pipeline.fit(X)
        results = pipeline.predict(X)
        self.assertEqual(results['SimpleEstimatorStep'], 3.2)

    def test_iterator(self):
        steps = [('SimpleOptimizerStep', SimpleOptimizerStep())]
        pipeline = QdPipeline(
            steps, global_iterator=SimpleIteratorStep(loop_num=2)
        )
        X = 10
        result = pipeline.optimize(X)
        self.assertEqual(result['SimpleOptimizerStep'], 26)

    def test_global_params(self):
        params = {'alpha': 0.1}
        steps = [('SimpleOptimizerStep', SimpleOptimizerStep())]
        pipeline = QdPipeline(
            steps, global_iterator=SimpleIteratorStep(loop_num=2)
        )
        pipeline.set_global_params(params)
        self.assertEqual(pipeline.get_global_params(), params)


if __name__ == '__main__':
    unittest.main()
