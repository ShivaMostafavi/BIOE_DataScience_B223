import sys

from ml.utils import get_experiment_data, get_estimator, run_cv, print_scores, cross_validation
sys.path.extend(['.', '..'])

experiment = {
  "classifier": "xgboost",
  "experiment": "a",
  "data": "movement",
  "mode": None,
  "channel_filter": "Both",
  "exp_name": "xg_compare"
}

hyperparams = {
  "xgboost": {
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [3, 6],
    "clf__learning_rate": [0.01, 0.1],
    "clf__subsample": [0.8, 1.0]
  }
}

X, y, c, channels = get_experiment_data(experiment, return_idxs=False)
estimator = get_estimator(experiment=experiment, hyperparams=hyperparams,
                          data_shape=X.shape[1:], gridsearch=True, root="./")
scores = run_cv(X, y, estimator, cross_validation, weighting_factor=c)  # 若不需权重传 None
print_scores(X, scores, to_console=True, to_file=True, file_name="xgboost_scores")