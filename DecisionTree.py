import numpy as np
import math
from abc import ABC, abstractmethod


class TreeNode(object):
    def __init__(self, value_voted=None, value_selected=None, feature_i=None,
                 left_true=None, right_wrong=None):
        self.value_voted = value_voted
        self.value_selected = value_selected
        self.feature_i = feature_i
        self.left_true = left_true
        self.right_wrong = right_wrong

class DecisionTree(ABC):
    def __init__(self, min_splitted=2, max_depth=10, min_impurity=1e-5):
        self.root = None
        self.min_splitted = min_splitted
        self.max_depth = max_depth
        self.min_impurity = min_impurity

    def fit(self, X, y, metric="gini_index"):
        if metric != "gini_index":
            if metric != "gain_ratio":
                metric = "gini_index"
        self.root = self._split_tree(X, y,metric)

    def _get_values_on_feature(self, X, feature, value):
        Xy1 = []
        Xy2 = []
        for sample in X:
            if isinstance(value, int) or isinstance(value, float):
                if sample[feature] >= value:
                    Xy1.append(sample)
                else:
                    Xy2.append(sample)
            else:
                if sample[feature] == value:
                    Xy1.append(sample)
                else:
                    Xy2.append(sample)
        return np.array(Xy1), np.array(Xy2)

    def _split_tree(self, X, y, metric, cur_depth=0):
        print('input', y)
        max_impurity = 0
        best_split = None
        best_sets = None
        no_samples, no_features = X.shape
        if len(y.shape) == 1:
            y = np.reshape(y, (y.shape[0], 1))
        print("concat", X)
        print("out", y)
        Xy = np.concatenate((X, y), axis=1)
        if no_samples >= self.min_splitted and cur_depth <= self.max_depth:
            for feature in range(no_features):
                feature_column = np.reshape(X[:, feature], (no_samples, 1))
                unique_values = np.unique(feature_column)
                for value in unique_values:
                    Xy1, Xy2 = self._get_values_on_feature(Xy, feature, value)
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, -1:]
                        y2 = Xy2[:, -1:]
                        impurity = self._calculate_impurity(y, y1, y2,metric)
                        if impurity > max_impurity:
                            max_impurity = impurity
                            best_split = {
                                "best_feature": feature,
                                "best_value": value 
                            }
                            best_sets = {
                                "left_set_X": Xy1[:, :-1],
                                "left_set_y": y1,
                                "right_set_X": Xy2[:, :-1],
                                "right_set_y": y2
                            } 
                            print('best_sets', best_sets)
        if max_impurity > self.min_impurity:
            left_true = self._split_tree(best_sets["left_set_X"], 
                                            best_sets["left_set_y"],metric, cur_depth+1)
            right_wrong = self._split_tree(best_sets["right_set_X"],
                                            best_sets["right_set_y"],metric, cur_depth+1)
            return TreeNode(feature_i=best_split["best_feature"], value_selected=best_split["best_value"],
                            left_true=left_true, right_wrong=right_wrong)

        return TreeNode(value_voted=self._calculate_leaf_value(y))

    def predict_value(self, x, tree=None):

        if tree is None:
            tree = self.root
        if tree.value_voted is not None:
            return tree.value_voted
        feature_value = x[tree.feature_i]
        branch = tree.right_wrong
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.value_selected:
                branch = tree.left_true
        elif feature_value == tree.value_selected:
            branch = tree.left_true

        return self.predict_value(x, branch)

    def predict(self, X):
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        """ Recursively print the decision tree """
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.value_voted is not None:
            print(tree.value_voted)
        # Go deeper down the tree
        else:
            # Print test
            print("%s==%s? " % (tree.feature_i, tree.value_selected))
            # Print the true scenario
            print("%sT->" % (indent), end="")
            self.print_tree(tree.left_true, indent + indent)
            # Print the false scenario
            print("%sF->" % (indent), end="")
            self.print_tree(tree.right_wrong, indent + indent)

    @abstractmethod
    def _calculate_impurity(self, y, y1, y2,metric):
        pass

    @abstractmethod
    def _calculate_leaf_value(self, y):
        pass

class ClassificationTree(DecisionTree):

    def _calculate_entropy(self, y):
        unique_y = np.unique(y)
        entropy = 0
        log2 = lambda x: math.log(x)/math.log(2)
        for value in unique_y:
            p = len(y[y==value])/len(y)
            entropy += -p*log2(p)
        return entropy

    def _calculate_gini(self, y):
        size = len(y)
        gini = 1
        # Return 0 impurity for the empty set
        if size == 0:
            return 0.0
        # Get counts of element values in array
        uniques, counts = np.unique(y, return_counts=True)
        # Calculate impurity = 1 - sum(squared_probability)
        for count in counts:
            gini += -(count / size) * (count / size)
        return gini

    def _calculate_impurity(self, y, y1 ,y2,metric):
        p = len(y1)/len(y)
        impurity_parent = 0
        impurity_child = 0

        if metric == 'gini_index':
            impurity_parent = self._calculate_gini(y)
            impurity_child = p * self._calculate_gini(y1) + (1 - p) * self._calculate_gini(y2)
        if metric == 'gain_ratio':
            impurity_parent = self._calculate_entropy(y)
            impurity_child = p*self._calculate_entropy(y1) + (1-p)*self._calculate_entropy(y2)
        gain = impurity_parent - impurity_child
        return gain

    def _calculate_leaf_value(self, y):
        max_no_value = 0
        most_common_value = None
        for label in np.unique(y):
            no_value = len(y[y==label])
            if no_value > max_no_value:
                max_no_value = no_value
                most_common_value = label
        return most_common_value


class RegressionTree(DecisionTree):

    def _calculate_variance(self, X):
        mean = np.ones(np.shape(X)) * X.mean(0)
        n_samples = np.shape(X)[0]
        variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
        return variance

    def _calculate_impurity(self, y, y1, y2):
        var_tot = self._calculate_variance(y)
        var_1 = self._calculate_variance(y1)
        var_2 = self._calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)
        return sum(variance_reduction)

    def _calculate_leaf_value(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]