"""
    Reference:
        [1] http://statweb.stanford.edu/~tibs/book/chap14
"""""

import numpy as np
from scipy.special import expit, logsumexp
from abc import ABCMeta, abstractmethod


class Loss(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, y, pred):
        """Compute the loss.

        Parameters
        ----------
        y : array, shape (n_samples,)
            True labels

        pred : array, shape (n_samples,)
            Predicted labels
            pred is a log odds
        """

    @abstractmethod
    def negative_gradient(self, y, pred):
        """Compute the negative gradient.

        Parameters
        ----------
        y : array, shape (n_samples)
            The target labels.

        pred : array, shape (n_samples)
            The predictions.
        """


# region Regression Losses
class LeastSquaresError(Loss):
    """Least Squares loss function for regression"""

    def __call__(self, y, y_pred):
        return np.mean((y - y_pred.ravel()) ** 2.0)

    def negative_gradient(self, y, y_pred):
        return y - y_pred.ravel()


class LeastAbsoluteError(Loss):
    """Least absolute loss function for regression"""

    def __call__(self, y, pred):
        return np.sum(np.abs(y - pred))

    def negative_gradient(self, y, pred):
        return np.sign(y - pred)
# endregion RegressionLosses


# region Classification Losses
class BinomialDeviance(Loss):
    """Binomial deviance loss function for binary classification.
    Logistic loss function (logloss), aka binomial deviance, aka cross-entropy, aka log-likelihood loss.

    Parameters
    ---------
    pred: is a log odds
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        super(Loss, self).__init__()

    def __call__(self, y, pred):
        """Compute the deviance (= 2 * negative log-likelihood)"""
        pred = pred.ravel()
        return -2.0 * np.mean((y * pred) - np.logaddexp(0.0, pred))

    def negative_gradient(self, y, pred):
        """Compute the residual (= negative gradient)"""
        return y - expit(pred.ravel())


class MultinomialDeviance(Loss):
    """Multinomial deviance loss function for multi-class classification.

        Parameters
        ----------
        n_classes : int
            Number of classes
        """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        super(Loss, self).__init__()

    def __call__(self, y, pred):
        """Compute the Multinomial deviance.
        This compute refer to equation (14.22) of Reference[1]
        """

        # create one-hot label encoding
        Y = np.zeros((y.shape[0], self.n_classes), dtype=np.float64)
        for k in range(self.n_classes):
            Y[:, k] = y == k

        return np.sum(-1 * (Y * pred).sum(axis=1) + logsumexp(pred, axis=1))

    def negative_gradient(self, y, pred, k=0):
        """Compute negative gradient for the ``k``-th class."""
        return y - np.nan_to_num(np.exp(pred[:, k] - logsumexp(pred, axis=1)))
# endregion ClassificationLosses
