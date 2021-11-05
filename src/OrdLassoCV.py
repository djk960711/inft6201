import numpy as np
from sklearn import linear_model, svm, metrics


class OrdinalLasso(linear_model.Lasso):
    """
    Overwrite Ridge from scikit-learn to use
    the (minus) absolute error as score function.

    (see https://github.com/scikit-learn/scikit-learn/issues/3848
    on why this cannot be accomplished using a GridSearchCV object)
    """

    def fit(self, X, y, **fit_params):
        self.unique_y_ = np.unique(y)
        super(linear_model.Lasso, self).fit(X, y, **fit_params)
        return self

    def predict(self, X):
        pred = np.round(super(linear_model.Lasso, self).predict(X))
        pred = np.clip(pred, 0, self.unique_y_.max())
        return pred

    def score(self, X, y):
        pred = self.predict(X)
        return metrics.mean_squared_error(pred, y)


from sklearn.base import clone


class OrdinalClassifier():

    def __init__(self, clf):
        self.clf = clf
        self.clfs = {}

    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = clone(self.clf)
                clf.fit(X, binary_y)
                self.clfs[i] = clf

    def predict_proba(self, X):
        clfs_predict = {k: self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i, y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:, 1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                predicted.append(clfs_predict[y - 1][:, 1] - clfs_predict[y][:, 1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y - 1][:, 1])
        return np.vstack(predicted).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)