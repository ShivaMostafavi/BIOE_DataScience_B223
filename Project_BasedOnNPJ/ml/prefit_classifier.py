from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels


class PrefitClassifier(BaseEstimator, ClassifierMixin):
    """
    A wrapper that is used to load prefit checkpoints instead of re-training when calling fit.
    Used in combination with the NeuralNetClassifier class from skorch and the respective checkpoint format.
    """
    def __init__(self, prefit_cp=None, net=None):
        self.classes_ = [0, 1]
        self.prefit_cp = prefit_cp
        self.net = net

    def fit(self, x, y, **ft_params):
        self.classes_ = unique_labels(y)
        self.net.initialize()
        self.net.load_params(f_params=self.prefit_cp)
        return self

    def predict(self, x):
        return self.net.predict(x)

    def predict_proba(self, x):
        return self.net.predict_proba(x)

    def set_params(self, **kwargs):
        self.net.set_params(**kwargs)
