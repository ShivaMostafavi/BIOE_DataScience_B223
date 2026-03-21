import json
import pandas as pd
from ml.utils import dataset_root, get_dataset, cross_validation, metrics
from utils.experiment import experiment
import numpy as np

df = pd.read_csv(dataset_root + "/out/quest_res_folds.csv", sep="\t")

with open(dataset_root + "/out/best_params_quest.json", "rb") as f:
    params_list = json.load(f)

with open(dataset_root + "/out/stratified_subset.json", "rb") as f:
    idxs_strat = json.load(f)
    idxs_strat = np.array(idxs_strat)

cv = cross_validation

all_res = []
counter = 0
for mode in ["pd_vs_dd", "pd_vs_hc"]:
    _, y, _, idxs = get_dataset(type="questionnaire", mode=mode, return_idxs=True)
    for test_fold, (train_idx, test_idx) in enumerate(cv.split(idxs, y)):
        _, idxs_strat_test, _ = np.intersect1d(idxs[test_idx], idxs_strat, return_indices=True)

        exp = df[(df["mode"] == mode) & (df["test_fold"] == test_fold+1)]
        exp = exp.iloc[0, :]
        params = {exp["classifier"]: params_list[counter]}
        _, pred = experiment(exp, params, dataset_root, test_fold)
        pred = pred[idxs_strat_test]

        res = {key: [] for key in metrics.keys()}
        for key, item in metrics.items():
            res[key].append(item(y[test_idx][idxs_strat_test], pred))
        res["mode"] = mode
        all_res.append(res)
        counter += 1

df = pd.DataFrame(all_res)
df.loc[:, ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]] = df.loc[:, ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]].map(lambda x: x[0])

df_grouped = df.groupby(["mode"])
df_mean = df_grouped[["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]].mean().astype(float)
df_std = df_grouped[["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]].std().astype(float)
df_mean = df_mean.apply(lambda col: (col * 100).round(2).astype(str)) + " (" + df_std.apply(lambda col: (col * 100).round(2).astype(str)) + ")"
df_mean.to_csv(f"{dataset_root}/out/quest_res_stratified.csv", index=False, sep="\t")
print("done")
