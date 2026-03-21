import numpy as np
import pandas as pd
from grouped_permutation_importance import grouped_permutation_importance
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from ml.input_selector import InputSelector, InputPacker
from ml.sample_weight_pipeline import SampleWeightPipeline
from ml.utils import get_estimator, get_experiment_data
from ml.sample_weighting import get_balanced_weights
from ml.utils import get_file_list, get_dataset, cross_validation, metrics, group_channels, get_channels


def run_stacking(experiment_list, hyperparams_list, root, test_fold=None):
    x_list = []
    for experiment in experiment_list:
        x, y, c, channels = get_experiment_data(experiment)
        x_list.append(x)
    x_len = len(x_list[0])
    x_new = []
    for idx in range(x_len):
        row = []
        for sub_x in x_list:
            row.append(sub_x[idx])
        x_new.append(row)
    x = np.array(x_new, dtype=object)

    cv = cross_validation

    data_shapes = [x[0][idx].shape for idx in range(len(x[0]))]

    # Prepare score
    score = {key: [] for key in metrics.keys()}

    for idx, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        if test_fold is not None:
            if idx != test_fold:
                continue
        weights = get_balanced_weights(y[train_idx], c[train_idx])
        all_estimators = []
        for n, experiment in enumerate(experiment_list):
            estimator = get_estimator(experiment, hyperparams=hyperparams_list[n], data_shape=data_shapes[n], idx=idx,
                                      root=root, gridsearch=False, prefit=True)
            feature_selection = InputSelector(idx=n)
            estimator.steps.insert(0, ("select", feature_selection))
            all_estimators.append((f"{n}", estimator))
        estimator = StackingClassifier(estimators=all_estimators,
                                       final_estimator=LogisticRegression(random_state=42),
                                       passthrough=False, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        estimator.fit(x[train_idx], y[train_idx], sample_weight=weights)
        pred = estimator.predict(x[test_idx])
        for key, item in metrics.items():
            score[key].append(item(y[test_idx], pred))

    return score, pred


def run_stacking_feat_imp(experiment_list, hyperparams_list, feature_importance_groups, root, test_fold=None):
    x_list = []
    modality_groups = []
    for experiment, (modality_name, modality_def) in zip(experiment_list, feature_importance_groups.items()):
        x, y, c, channels = get_experiment_data(experiment)
        x_list.append(x)
        imp_groups = []
        # group_idxs_len = np.prod(x.shape[1:])
        # group_idxs = np.full(group_idxs_len, False)
        for group_name, group_def in modality_def.items():
            group_filter = "|".join(group_def)
            group_idxs = pd.Series(channels).str.contains(group_filter).values
            if len(x.shape) > 2:
                group_idxs = np.repeat(group_idxs, x.shape[-1])
            # group_size = np.sum(group_idxs)
            imp_groups.append(group_idxs)
        modality_groups.append(imp_groups)

    group_idxs_lens = []
    for modality_group in modality_groups:
        group_idxs_lens.append(len(modality_group[0]))

    all_group_idxs = []
    offset = 0
    for modality_group, group_idx_len in zip(modality_groups, group_idxs_lens):
        for group_idxs in modality_group:
            full_group_idxs = np.full(np.sum(group_idxs_lens), False)
            full_group_idxs[offset:offset+group_idx_len] = group_idxs
            all_group_idxs.append(full_group_idxs)
        offset += group_idx_len

    data_shapes = []
    for x_next in x_list:
        data_shapes.append(x_next.shape[1:])

    x_len = len(x_list[0])
    x_new = []
    for idx in range(x_len):
        row = []
        for sub_x in x_list:
            row.append(sub_x[idx].flatten())
        row = np.concatenate(row)
        x_new.append(row)
    x = np.array(x_new)

    cv = cross_validation

    # Prepare score
    score = {key: [] for key in metrics.keys()}

    imps = []
    coefs = []
    for idx, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        if test_fold is not None:
            if idx != test_fold:
                continue
        weights = get_balanced_weights(y[train_idx], c[train_idx])
        all_estimators = []
        for n, experiment in enumerate(experiment_list):
            estimator = get_estimator(experiment, hyperparams=hyperparams_list[n], data_shape=data_shapes[n], idx=idx,
                                      root=root, gridsearch=False, prefit=True)
            feature_selection = InputSelector(idx=n)
            estimator.steps.insert(0, ("select", feature_selection))
            all_estimators.append((f"{n}", estimator))
        estimator = StackingClassifier(estimators=all_estimators, n_jobs=-1,
                                       final_estimator=LogisticRegression(random_state=42),
                                       passthrough=False, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        estimator = SampleWeightPipeline([("packer", InputPacker(specs=data_shapes)), ("clf", estimator)])
        estimator.fit(x[train_idx], y[train_idx], sample_weight=weights)
        pred = estimator.predict(x[test_idx])
        for key, item in metrics.items():
            score[key].append(item(y[test_idx], pred))
        imp = grouped_permutation_importance(estimator, x[test_idx], y[test_idx], idxs=all_group_idxs,
                                             n_repeats=20, random_state=42, scoring="balanced_accuracy", n_jobs=5,
                                             cv=None, perm_set=None)
        imp = imp["importances"].tolist()
        imps.append(imp)
        coefs.append(estimator["clf"].final_estimator_.coef_[0].tolist())

    return score, pred, imps, coefs
