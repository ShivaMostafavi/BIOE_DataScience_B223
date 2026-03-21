import json
import numpy as np
import pandas as pd
from ml.utils import dataset_root, get_file_list, cross_validation, get_dataset, metrics

file_list = get_file_list()
cv = cross_validation

with open(dataset_root + "/out/stratified_subset.json", "rb") as f:
    idxs_strat = json.load(f)
    idxs_strat = np.array(idxs_strat)

with open(dataset_root + "/out/stacking_res.json", "rb") as f:
    best_quest = json.load(f)

all_res = []
counter = 0
for mode in ["pd_vs_dd", "pd_vs_hc"]:
    _, y, _, idxs = get_dataset(type="questionnaire", mode=mode, return_idxs=True)
    for test_fold, (train_idx, test_idx) in enumerate(cv.split(idxs, y)):
        pred = np.array(best_quest[counter]["pred"])
        _, idxs_strat_test, _ = np.intersect1d(idxs[test_idx], idxs_strat, return_indices=True)

        pred = pred[idxs_strat_test]

        score = {key: [] for key in metrics.keys()}
        for key, item in metrics.items():
            score[key].append(item(y[test_idx][idxs_strat_test], pred))
        score["mode"] = mode
        all_res.append(score)
        counter += 1

df = pd.DataFrame(all_res)
df.loc[:, ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]] = df.loc[:, ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]].map(lambda x: x[0])

df_grouped = df.groupby(["mode"])
df_mean = df_grouped[["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]].mean().astype(float)
df_std = df_grouped[["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]].std().astype(float)
df_mean = df_mean.apply(lambda col: (col * 100).round(2).astype(str)) + " (" + df_std.apply(lambda col: (col * 100).round(2).astype(str)) + ")"
df_mean.to_csv(f"{dataset_root}/out/stacking_res_stratified.csv", index=False, sep="\t")
print("done")