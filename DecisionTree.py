class Node(object):
    def __init__(self, data=None, left=None, right= None):
        # TO DO
        self.left = left
        self.right = right
        self.data = data


class DecisionTree(object):
    def __init__(self, max_depth=6, min_split=2):
        # TO DO
        self.max_depth = max_depth
        self.min_split = min_split


    def fit(self, X, y):
        #TO DO
        pass

    def predict(self):
        #TO DO
        pass


class RegressionTree(DecisionTree):
    # TO DO
    pass


class ClassificationTree(DecisionTree):
    # TO DO
    pass
