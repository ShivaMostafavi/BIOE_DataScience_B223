import sys

from ml.utils import get_experiment_data, get_estimator, run_cv, print_scores, cross_validation
sys.path.extend(['.', '..'])

experiment = {
  "classifier": "svm",
  "experiment": "b",
  "data": "movement",
  "mode": "none",
  "channel_filter": "Both",
  "exp_name": "svm_compare_pd_vs_dd_hc"
}

hyperparams = {
  "svm": {
    "clf__C": [0.1, 1.0, 10.0],
    "clf__kernel": ["linear", "rbf"],
    "clf__gamma": ["scale", "auto"]
  }
}

X, y, c, channels = get_experiment_data(experiment, return_idxs=False)
estimator = get_estimator(experiment=experiment, hyperparams=hyperparams,
                          data_shape=X.shape[1:], gridsearch=True, root="./")
scores = run_cv(X, y, estimator, cross_validation, weighting_factor=c)
print_scores(X, scores, to_console=True, to_file=True, file_name="svm_scores")
