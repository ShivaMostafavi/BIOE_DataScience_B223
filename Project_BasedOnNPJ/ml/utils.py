from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score, \
    balanced_accuracy_score
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
from ml.sample_weight_pipeline import SampleWeightPipeline
from ml.feature_extraction import feature_extraction
from ml.sample_weighting import get_balanced_weights
import torch
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from skorch.callbacks import Checkpoint
from ml.multi_boss import MultiBOSS
from ml.nn import get_nn
from ml.prefit_classifier import PrefitClassifier
from ml.xception import get_xception

# Dataset directory
dataset_root = 'D:\\A_UCLA\\26Win_Classes\\223B\\Project\\pads-parkinsons-disease-smartwatch-dataset-1.0.0\\pads-parkinsons-disease-smartwatch-dataset-1.0.0\\preprocessed' #"./data/"
file_list_name = "file_list.csv"
# Define the metrics that are reported
metrics = {"accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score,
           "f1": f1_score, "precision": precision_score, "recall": recall_score, "roc_auc": roc_auc_score}
# Define the cv
cross_validation = RepeatedStratifiedKFold(n_repeats=1, n_splits=5, random_state=42)


def get_channels(type="movement"):
    channels = []
    if type == "movement":
        for task in ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold", "HoldWeight",
                     "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1", "Entrainment2"]:
            for device_location in ["LeftWrist", "RightWrist"]:
                for sensor in ["Acceleration", "Rotation"]:
                    for axis in ["X", "Y", "Z"]:
                        channel = f"{task}_{sensor}_{device_location}_{axis}"
                        channels.append(channel)
    elif type == "questionnaire":
        channels = [f"NMS_{n:02d}" for n in range(1, 31)]
    return channels


def get_channel_name_part(ch_names, by="suffix", sep="_", pos=1):
    all_parts = []
    for idx, ch_name in enumerate(ch_names):
        ch_name_split = ch_name.split(sep)
        ch_name_part = sep.join(ch_name_split[pos:])
        if ch_name_part not in all_parts:
            all_parts.append(ch_name_part)
    return all_parts


def get_channel_idx(ch_names, name=None, remove=False):
    ch_idxs = []
    for idx, ch_name in enumerate(ch_names):
        if name in ch_name:
            ch_idxs.append(idx)
    return ch_idxs


def group_channels(data, ch_names, by="suffix", sep="_", pos=2):
    all_parts = get_channel_name_part(ch_names, by=by, sep=sep, pos=pos)
    new_data = []
    for part in all_parts:
        ch_idxs = get_channel_idx(ch_names, name=part, remove=False)
        new_data.append([data_point for n in ch_idxs for data_point in data[n]])
    return np.array(new_data)


def get_file_list(mode=None):
    file_list = pd.read_csv(dataset_root + "/file_list.csv")
    if mode == "pd_vs_hc":
        file_list = file_list.loc[(file_list["label"] == 0) | (file_list["label"] == 1)]
    if mode == "pd_vs_dd":
        file_list = file_list.loc[(file_list["label"] == 1) | (file_list["label"] == 2)]
        file_list["label"].replace({1: 0, 2: 1}, inplace=True)
    return file_list


def get_dataset(type="movement", mode=None, channel_filter=None, feature_extraction=None, return_idxs=False):
    channels = pd.Series(get_channels(type))
    file_list = get_file_list(mode)
    y = file_list["label"].values
    print("Load dataset")
    subfolder = ""
    if type == "movement":
        subfolder = "movement/"
    elif type == "questionnaire":
        subfolder = "questionnaire/"
    x = []
    for idx, file_idx in enumerate(file_list["id"]):
        # Print progress
        if idx % 100 == 0:
            print(f"{idx + 1} / {len(file_list)}")
        if type == "movement":
            data = np.fromfile(f"{dataset_root}/{subfolder}{file_idx:03d}_ml.bin", dtype=np.float32).reshape((-1, 976))
        elif type == "questionnaire":
            data = np.fromfile(f"{dataset_root}/{subfolder}{file_idx:03d}_ml.bin", dtype=np.float32).reshape((30))
        if channel_filter is not None:
            channel_mask = channels.str.contains(channel_filter)
            data = data[channel_mask]
            new_channels = channels[channel_mask]
        else:
            new_channels = channels
        new_channels = new_channels.values
        if feature_extraction is not None:
            data = feature_extraction(data)
            n_feature_per_channel = data.shape[0] // len(new_channels)
            new_channels = np.repeat(new_channels, n_feature_per_channel)
        x.append(data)
    x = np.stack(x)
    if return_idxs:
        idxs = file_list.index.values
        return x, y, new_channels, idxs
    return x, y, new_channels


def print_scores(data, scores, to_console=True, to_file=True, file_name="score"):
    if to_console:
        print(f"Data size: {data.shape}")
        for key, item in scores.items():
            if len(item) > 1:
                print(f"{key}: {np.mean(item):.4f} +/- {np.std(item):.4f}")
            else:
                print(f"{key}: {item[0]:.4f}")
    if to_file:
        with open(f"{file_name}.txt", "w") as f:
            print(f"Data size: {data.shape}", file=f)
            for key, item in scores.items():
                if len(item) > 1:
                    print(f"{key}: {np.mean(item):.4f} +/- {np.std(item):.4f}", file=f)
                else:
                    print(f"{key}: {item[0]:.4f}", file=f)


def _get_classification_average(labels):
    return "binary" if np.unique(labels).shape[0] == 2 else "weighted"


def _get_score_input(estimator, data):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(data)
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(data)
    return None


def _compute_metric(metric_name, y_true, y_pred, score_input=None):
    average = _get_classification_average(y_true)
    if metric_name == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric_name == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if metric_name == "f1":
        return f1_score(y_true, y_pred, average=average, zero_division=0)
    if metric_name == "precision":
        return precision_score(y_true, y_pred, average=average, zero_division=0)
    if metric_name == "recall":
        return recall_score(y_true, y_pred, average=average, zero_division=0)
    if metric_name == "roc_auc":
        try:
            if score_input is None:
                return roc_auc_score(y_true, y_pred) if average == "binary" else np.nan
            if average == "binary":
                if np.ndim(score_input) == 2:
                    if score_input.shape[1] == 2:
                        score_input = score_input[:, 1]
                    elif score_input.shape[1] == 1:
                        score_input = score_input[:, 0]
                return roc_auc_score(y_true, score_input)
            if np.ndim(score_input) != 2:
                return np.nan
            if score_input.shape[1] != np.unique(y_true).shape[0]:
                return np.nan
            class_labels, class_counts = np.unique(y_true, return_counts=True)
            aucs = []
            for class_idx, class_label in enumerate(class_labels):
                aucs.append(roc_auc_score((y_true == class_label).astype(int), score_input[:, class_idx]))
            return np.average(aucs, weights=class_counts)
        except ValueError:
            return np.nan
    raise KeyError(f"Unknown metric: {metric_name}")


def run_cv(data, labels, estimator, cv, weighting_factor=None):
    score = {key: [] for key in metrics.keys()}
    for idx, (train_idx, test_idx) in enumerate(cv.split(data, labels)):
        if weighting_factor is not None:
            weights = get_balanced_weights(labels[train_idx], weighting_factor[train_idx])
        else:
            weights = get_balanced_weights(labels[train_idx], None)
        estimator.fit(data[train_idx, :], labels[train_idx], clf__sample_weight=weights)
        pred = estimator.predict(data[test_idx, :])
        score_input = _get_score_input(estimator, data[test_idx, :])
        for key in metrics.keys():
            score[key].append(_compute_metric(key, labels[test_idx], pred, score_input=score_input))
    return score


def get_estimator(experiment=None, hyperparams=None, data_shape=None, idx=None, root=None, gridsearch=True,
                  prefit=False):
    # Safely get hyperparameter search space for the classifier (may be None)
    if hyperparams is None:
        params = {}
    else:
        params = hyperparams.get(experiment["classifier"], {})
    pipeline_elems = []
    if experiment["experiment"] == "b":
        pipeline_elems.append(
            ("boss", MultiBOSS(data_shape, window_sizes=[20, 40, 80], window_step=2, buf_path=f"{root}out/")))
    if experiment["classifier"] == "nn":
        if experiment["experiment"] == "b":
            data_shape = None
        else:
            data_shape = data_shape[0]
        cp = Checkpoint(dirname=f"{root}out/cp_{experiment['exp_name']}/{idx:02d}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        estimator = get_nn(data_shape, cp, device)
        if prefit:
            estimator = PrefitClassifier(prefit_cp=f"{root}out/cp_{experiment['exp_name']}/{idx:02d}/params.pt",
                                         net=estimator)
        pipeline_elems.extend([("scaler", StandardScaler()),
                               ("clf", estimator)])
        estimator = SampleWeightPipeline(pipeline_elems)
    elif experiment["classifier"] == "svm":
        pipeline_elems.extend([("scaler", StandardScaler()),
                               ("clf", SVC(probability=False, verbose=False))])
        estimator = SampleWeightPipeline(pipeline_elems)
    elif experiment["classifier"] == "cat":
        pipeline_elems.extend([("scaler", 'passthrough'),
                               ("clf", CatBoostClassifier(random_state=42, verbose=False))])
        estimator = SampleWeightPipeline(pipeline_elems)
    elif experiment["classifier"] == "xgboost":
        pipeline_elems.extend([("scaler", StandardScaler()),
                               ("clf", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
        estimator = SampleWeightPipeline(pipeline_elems)
    elif experiment["classifier"] == "xception":
        data_shape = data_shape[0]
        cp = Checkpoint(dirname=f"{root}out/cp_{experiment['exp_name']}/{idx:02d}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"
        estimator = get_xception(data_shape, cp, device)
        if prefit:
            estimator = PrefitClassifier(prefit_cp=f"{root}out/cp_{experiment['exp_name']}/{idx:02d}/params.pt",
                                         net=estimator)
    else:
        raise Exception("classifier not specified correctly")
    if gridsearch:
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        estimator = GridSearchCV(estimator, params, cv=inner_cv, scoring="balanced_accuracy", verbose=1)
    else:
        estimator.set_params(**params)

    return estimator


def get_experiment_data(experiment, return_idxs=False):
    c = get_file_list(mode=experiment["mode"])["gender"].values
    channel_filter = experiment["channel_filter"]
    if channel_filter == "Both":
        channel_filter = None
    feature = None
    if experiment["experiment"] == "a":
        feature = feature_extraction
    if return_idxs:
        x, y, channels, idxs = get_dataset(type=experiment["data"], mode=experiment["mode"],
                                           channel_filter=channel_filter, feature_extraction=feature,
                                           return_idxs=return_idxs)
        return x, y, c, channels, idxs
    x, y, channels = get_dataset(type=experiment["data"], mode=experiment["mode"], channel_filter=channel_filter,
                                 feature_extraction=feature)
    return x, y, c, channels
