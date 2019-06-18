import numpy as np
from DecisionTree import RegressionTree
from loss_functions import LeastSquaresError, BinomialDeviance


class GradientBoosting(object):
    """Super class for Gradient Boosting"""

    def __init__(self, loss, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, regression):
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression

        # Check parameter
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0")

        if self.regression:
            if self.loss == 'ls':
                pass
        elif self.loss == 'deviance':
            pass
        else:
            raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))

    def fit(self, X, y):
        """Fit the gradient boosting model."""

        if self.regression:
            # regression
            if self.loss == 'ls':
                self.loss_func = LeastSquaresError()
            self.loss_func.is_multi_class = False
        else:
            # classification
            n_classes = np.unique(y).shape[0]
            if n_classes == 2:
                if self.loss == 'deviance':
                    self.loss_func = BinomialDeviance(n_classes=1)
            else:
                self.loss_func.is_multi_class = True
                self.loss_func.n_classes = n_classes

        # Initialize the model
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))

        # fit the boosting stages
        estimators = []
        for i in range(self.n_estimators):

            # induce regression tree on residuals
            tree = RegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity=self.min_impurity,
                max_depth=self.max_depth)

            residual = self.loss_func.negative_gradient(y, y_pred)

            # Fit base learner to pseudo residuals
            learner = tree.fit(X, residual)

            # update tree leaves
            update = learner.predict(X)
            y_pred -= np.multiply(self.learning_rate, update)

            # add tree to ensemble
            estimators.append(learner)
        self.estimators = estimators

    def predict(self, X):
        """An estimator predicting the test data."""
        y_pred = np.zeros(np.shape(self.estimators[0].predict(X)), dtype=np.float64)
        for i in range(self.n_estimators):
            y_pred =self.estimators[i].predict(X)
            y_pred1 = np.add(y_pred,self.estimators[i].predict(X))
            print(y_pred)
        return y_pred
        #return np.sum(m.predict(X) for m in self.estimators[self.n_estimators])


class GradientBoostingRegression(GradientBoosting):

    def __init__(self, loss='ls', n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_var_red=1e-7, max_depth=4):
        super(GradientBoostingRegression, self).__init__(loss=loss,
                                                         n_estimators=n_estimators,
                                                         learning_rate=learning_rate,
                                                         min_samples_split=min_samples_split,
                                                         min_impurity=min_var_red,
                                                         max_depth=max_depth,
                                                         regression=True)


class GradientBoostingClassifier(GradientBoosting):

    def __init__(self, loss='deviance', n_estimators=200, learning_rate=.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2):
        super(GradientBoostingClassifier, self).__init__(loss=loss,
                                                         n_estimators=n_estimators,
                                                         learning_rate=learning_rate,
                                                         min_samples_split=min_samples_split,
                                                         min_impurity=min_info_gain,
                                                         max_depth=max_depth,
                                                         regression=False)

    """def fit(self, X, y):
    #    y = to_categorical(y)
    #    super(GradientBoostingClassifier, self).fit(X, y)
    """
