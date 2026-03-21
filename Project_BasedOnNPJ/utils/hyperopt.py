import numpy as np
from skorch.helper import SliceDataset
from ml.dl_dataset import DLDataset
from ml.utils import get_estimator, get_experiment_data
from ml.sample_weighting import get_balanced_weights
from ml.utils import cross_validation, metrics, group_channels


def hyperopt(experiment, hyperparams, root):
    """
    Run hyperparameter optimization for the defined experimental setup.
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
    test_score = {key: [] for key in metrics.keys()}
    best_params = []
    best_scores = []

    for idx, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        weights = get_balanced_weights(y[train_idx], c[train_idx])
        estimator = get_estimator(experiment, hyperparams=hyperparams, data_shape=data_shape, idx=idx, root=root)
        if experiment['classifier'] == 'xception':
            estimator.fit(x[train_idx], y[train_idx], sample_weight=weights)
        else:
            estimator.fit(x[train_idx], y[train_idx], clf__sample_weight=weights)
        best_params.append(estimator.best_params_)
        best_scores.append(estimator.best_score_)
        if (experiment['classifier'] == 'xception') and (experiment['experiment'] == 'c'):
            # Test whithout augmentation
            estimator.best_estimator_.load_params(f_params=f'{root}out/cp_{experiment["exp_name"]}/{idx:02d}/params.pt')
            pred = estimator.predict(x_test[test_idx])
        else:
            pred = estimator.predict(x[test_idx])
        for key, item in metrics.items():
            test_score[key].append(item(y[test_idx], pred))

    return {'test_score': test_score, 'score': best_scores, 'params': best_params}
