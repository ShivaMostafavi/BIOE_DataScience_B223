from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class InputSelector(BaseEstimator, TransformerMixin):
    """
    Selects data based on indices.
    """
    def __init__(self, idx=None):
        self.idx = idx

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y, **kwargs):
        return self

    def transform(self, X):
        if self.idx is None:
            return X
        out = X[:, self.idx]
        out = out.tolist()
        out = np.array(out)
        return out


class InputPacker(BaseEstimator, TransformerMixin):
    """
    Packs data based on specifications.
    """
    def __init__(self, specs=None):
        # specs = [(66, 976), (30,)]
        self.specs = specs

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y, **kwargs):
        return self

    def transform(self, X):
        if self.specs is None:
            return X
        out = []
        for elem in X:
            new_elem = []
            idx = 0
            for spec in self.specs:
                spec_len = np.prod(spec)
                shaped_elem = elem[idx:idx + spec_len]
                shaped_elem = np.reshape(shaped_elem, spec)
                new_elem.append(shaped_elem)
                idx += spec_len
            out.append(new_elem)
        out = np.array(out, dtype=object)
        return out
