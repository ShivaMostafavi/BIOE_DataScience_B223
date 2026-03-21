import numpy as np
from skorch.helper import SliceDataset
from ml.dl_dataset import DLDataset
from ml.utils import get_estimator, get_experiment_data
from ml.sample_weighting import get_balanced_weights
from ml.utils import cross_validation, metrics, group_channels


def experiment(experiment, hyperparams, root, test_fold=None):
    """
    Run a specific classification setup. Can only be performed after hyperparameter optimization.
    """
    x, y, c, channels = get_experiment_data(experiment)
    cv = cross_validation

    if experiment['experiment'] == 'c':
        x_new = []
        for x_sample in x:
            x_new.append(group_channels(x_sample, channels))
        x = np.array(x_new)
        x_test = DLDataset(mode='test', data=x, labels=y)
        x_test = SliceDataset(x_test, idx=0)
        x = DLDataset(mode='train', data=x, labels=y)
        x = SliceDataset(x, idx=0)

    data_shape = x[0].shape

    # Prepare score
    score = {key: [] for key in metrics.keys()}
    pred = None

    for idx, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        if test_fold is not None:
            if idx != test_fold:
                continue
        weights = get_balanced_weights(y[train_idx], c[train_idx])
        estimator = get_estimator(experiment, hyperparams=hyperparams, data_shape=data_shape, idx=idx, root=root,
                                  gridsearch=False, prefit=True)
        if experiment['classifier'] == 'xception':
            estimator.fit(x[train_idx], y[train_idx], sample_weight=weights)
        else:
            estimator.fit(x[train_idx], y[train_idx], clf__sample_weight=weights)
        pred = estimator.predict(x[test_idx])
        for key, item in metrics.items():
            score[key].append(item(y[test_idx], pred))

    return score, pred
