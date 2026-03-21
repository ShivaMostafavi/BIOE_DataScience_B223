import json
import numpy as np
import pandas as pd
from ml.stacking import run_stacking
from ml.utils import dataset_root

df_mov = pd.read_csv(dataset_root + "/out/mov_res_folds.csv", sep="\t")
df_quest = pd.read_csv(dataset_root + "/out/quest_res_folds.csv", sep="\t")

with open(dataset_root + "/out/best_params_mov.json", "rb") as f:
    params_list_mov = json.load(f)
with open(dataset_root + "/out/best_params_quest.json", "rb") as f:
    params_list_quest = json.load(f)

all_res = []
counter = 0
for mode in ["pd_vs_dd", "pd_vs_hc"]:
    for test_fold in range(1, 6):
        exp_list = []
        params_list = []
        exp = df_mov[(df_mov["mode"] == mode) & (df_mov["test_fold"] == test_fold)]
        exp = exp.iloc[0, :]
        params = {exp["classifier"]: params_list_mov[counter]}
        params_list.append(params)
        exp_list.append(exp)

        exp = df_quest[(df_quest["mode"] == mode) & (df_quest["test_fold"] == test_fold)]
        exp = exp.iloc[0, :]
        params = {exp["classifier"]: params_list_quest[counter]}
        params_list.append(params)
        exp_list.append(exp)
        res, pred = run_stacking(exp_list, params_list, dataset_root, test_fold-1)
        res["pred"] = pred.tolist()
        res["mode"] = mode
        all_res.append(res)
        counter += 1

with open(f"{dataset_root}/out/stacking_res.json", "w") as f:
    json.dump(all_res, f)

df = pd.DataFrame(all_res)
df.loc[:, ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]] = df.loc[:, ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]].map(lambda x: x[0])
df.to_csv(f"{dataset_root}/out/stacking_res_folds.csv", index=False, sep="\t")

df_grouped = df.groupby(["mode"])
df_mean = df_grouped[["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]].mean().astype(float)
df_std = df_grouped[["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]].std().astype(float)
df_mean = df_mean.apply(lambda col: (col * 100).round(2).astype(str)) + " (" + df_std.apply(lambda col: (col * 100).round(2).astype(str)) + ")"
df_mean.to_csv(f"{dataset_root}/out/stacking_res.csv", index=False, sep="\t")
print("done")