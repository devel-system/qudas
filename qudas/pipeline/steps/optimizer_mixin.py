from abc import ABC, abstractmethod
from .base import BaseStep

class OptimizerMixin(BaseStep, ABC):
    """
    Mixin class for all optimizers in qudas.
    This mixin supports `transform` and `optimize`, with `optimize` being required.
    """

    def transform(self, X):
        """
        Optional transform method. Can be overridden in subclasses.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        X_new : Transformed data.
        """
        return X  # デフォルトでは変換しない

    @abstractmethod
    def optimize(self, X=None, y=None, **fit_params):
        """
        Abstract method to be implemented in subclasses.
        Fits to data and returns optimized results.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        Optimized result.
        """
        pass
