from sklearn.pipeline import Pipeline


class SampleWeightPipeline(Pipeline):
    """
    A wrapper for the scikit-learn Pipeline. Passes the parameter sample_weight to the classifier in the pipeline.
    The classifier has to named 'clf'.
    """
    def fit(self, X, y=None, sample_weight=None, **fit_params):
        if sample_weight is not None:
            super().fit(X, y, clf__sample_weight=sample_weight, **fit_params)
        else:
            super().fit(X, y, **fit_params)
