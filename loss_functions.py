import numpy as np
from scipy.special import expit
from abc import ABCMeta,abstractmethod

class Loss(metaclass=ABCMeta):
    is_multi_class = False
    def __init__(self, n_classes):
        self.K = n_classes

    @abstractmethod
    def __call__(self, y, pred, sample_weight=None):
        """Compute the loss.

        Parameters
        ----------
        y : array, shape (n_samples,)
            True labels

        pred : array, shape (n_samples,)
            Predicted labels

        sample_weight : array-like, shape (n_samples,), optional
            Sample weights.
        """

    @abstractmethod
    def negative_gradient(self, y, y_pred):
        """Compute the negative gradient.

        Parameters
        ----------
        y : array, shape (n_samples,)
            The target labels.

        y_pred : array, shape (n_samples,)
            The predictions.
        """


class LeastSquaresError(Loss):
    """Least Squares loss function for regression"""

    def __call__(self, y, y_pred, sample_weight=None):
        if sample_weight is None:
            return np.mean((y - y_pred.ravel()) ** 2.0)
        else:
            return (1.0 / sample_weight.sum() *
                    np.sum(sample_weight * ((y - y_pred.ravel()) ** 2.0)))

    def negative_gradient(self, y, y_pred):
        return y - y_pred.ravel()
    def hessian

class BinomialDeviance(Loss):
    """Binomial deviance loss function for binary classification."""

    def __call__(self, y, pred, sample_weight=None):
        """Compute the deviance (= 2 * negative log-likelihood)"""
        pred = pred.ravel()
        return -2.0 * np.mean((y * pred) - np.logaddexp(0.0, pred))

    def negative_gradient(self, y, pred):
        """Compute the residual (= negative gradient)"""
        return y - expit(pred.ravel())
