import pickle
from pyts.transformation import BOSS
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np
import hashlib
from pathlib import Path


class MultiBOSS(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    Boss that is trained and performed per channel.
    """
    def __init__(self, data_shape=None, window_sizes=(40,), window_step=2, word_size=2, n_bins=3, buf_path="./"):
        self.classes_ = [0, 1]
        self.window_sizes = window_sizes
        self.window_step = window_step
        self.word_size = word_size
        self.n_bins = n_bins
        self.data_shape = data_shape
        self.buf_path = buf_path

        Path(self.buf_path + "boss/").mkdir(parents=True, exist_ok=True)

        self.boss_list = []
        # One BOSS per channel
        for _ in range(self.data_shape[0]):
            for window_size in self.window_sizes:
                self.boss_list.append(BOSS(sparse=False, window_size=window_size, window_step=self.window_step,
                                           word_size=self.word_size, n_bins=self.n_bins, norm_std=False,
                                           norm_mean=False, anova=False, drop_sum=False))

    def get_hash_path(self, X):
        s = X.tostring()
        h = hashlib.md5(s).hexdigest()
        h += "_"
        h += "_".join([str(wind_size) for wind_size in self.window_sizes])
        h += "_"
        h += "_".join(str(self.word_size))
        h += "_"
        h += "_".join(str(self.n_bins))
        h = self.buf_path + "boss/" + h + ".npy"
        return h

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y, **kwargs):
        # X = X.reshape((X.shape[0], self.data_shape[0], self.data_shape[1]))
        h = self.get_hash_path(X)
        if Path(h).is_file():
            with open(h, "rb") as f:
                print("Loading MultiBOSS")
                self.boss_list = pickle.load(f)
            return self
        boss_idx = 0
        for ch_idx in range(self.data_shape[0]):
            for _ in self.window_sizes:
                self.boss_list[boss_idx].fit(X[:, ch_idx, :], y)
                boss_idx += 1
        boss_list = self.boss_list
        with open(h, "wb") as f:
            print("Storing MultiBOSS")
            pickle.dump(boss_list, f)
        return self

    def transform(self, X):
        X_out = []
        boss_idx = 0
        for ch_idx in range(self.data_shape[0]):
            for _ in self.window_sizes:
                X_out.append(self.boss_list[boss_idx].transform(X[:, ch_idx, :]))
                boss_idx += 1
        X_out = np.concatenate(X_out, axis=1)
        X_out = X_out.astype(np.int32)
        return X_out

    def predict(self, X):
        return self.transform(X)

    def predict_proba(self, X):
        return self.transform(X)

    def set_params(self, **params):
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        self.boss_list = []
        # One BOSS per channel
        for _ in range(self.data_shape[0]):
            for window_size in self.window_sizes:
                self.boss_list.append(
                    BOSS(sparse=False, window_size=window_size, window_step=self.window_step, word_size=self.word_size,
                         n_bins=self.n_bins, norm_std=False, norm_mean=False, anova=False, drop_sum=False))
        return self
