from abc import ABC, abstractmethod
from .base import BaseStep

class IteratorMixin(BaseStep, ABC):
    """
    Mixin class for all iterators in qudas.
    This mixin requires `next_params` to be implemented.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loop_num = 1  # デフォルトで1回のループ
        self.models = None
        self.results = None

    @abstractmethod
    def next_params(self, X, y=None, **iter_params):
        """
        Abstract method to be implemented in subclasses.
        Generates next set of parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        **iter_params : dict
            Additional iteration parameters.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features_new)
            Transformed data with next set of parameters.
        """
        pass
