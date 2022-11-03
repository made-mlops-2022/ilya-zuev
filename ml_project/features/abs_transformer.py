import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AbsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.abs(X)
