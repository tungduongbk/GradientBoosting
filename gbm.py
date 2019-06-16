import numpy as np
from DecisionTree import RegressionTree
from data_manipulation import train_test_split, standardize, to_categorical
from data_operation import mean_squared_error, accuracy_score
from loss_functions import LeastSquaresError, BinomialDeviance

class GradientBoosting(object):
    """Super class for Gradient Boosting"""

     def __init__(self, loss ,n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, regression):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression
        self.loss = loss

        # Check parameter
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0")

        if self.loss == 'ls':
            self.loss_func = LeastSquaresError()
        elif self.loss == 'deviance':
            self.loss_func = BinomialDeviance()
        else:
            raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))

        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0,1] but "
                             "was %r" % self.subsample)

        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0.0, 1.0)")

    def fit(self, X, y):
        """Fit the gradient boosting model."""

        self.estimators = []

        # Initialize the model
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))

        # fit the boosting stages
        for i in range(self.n_estimators):

            # induce regression tree on residuals
            tree = RegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity=self.min_impurity,
                max_depth=self.max_depth)

            residual = self.loss_func.negative_gradient(y, y_pred)

            # Fit base learner to pseudo residuals
            tree.fit(X, residual)

            # update tree leaves
            update = tree.predict(X)
            y_pred -= np.multiply(self.learning_rate, update)

            # add tree to ensemble
            self.estimators.append(tree)

    def predict(self, X):
        """An estimator predicting the test data."""

        return np.sum(m.predict(X) for m in self.estimators)

class GradientBoostingRegressor(GradientBoosting):

    def __init__(self,loss='ls', n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_var_red=1e-7, max_depth=4, debug=False):
        super(GradientBoostingRegressor, self).__init__(loss=loss,
                                                        n_estimators=n_estimators,
                                                        learning_rate=learning_rate,
                                                        min_samples_split=min_samples_split,
                                                        min_impurity=min_var_red,
                                                        max_depth=max_depth,
                                                        regression=True)

class GradientBoostingClassifier(GradientBoosting):

    def __init__(self,loss ='deviance', n_estimators=200, learning_rate=.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2, debug=False):
        super(GradientBoostingClassifier, self).__init__(loss=loss,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_impurity=min_info_gain,
            max_depth=max_depth,
            regression=False)

    def fit(self, X, y):
        y = to_categorical(y)
        super(GradientBoostingClassifier, self).fit(X, y)