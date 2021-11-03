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
