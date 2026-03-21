import json
import pandas as pd
from ml.stacking import run_stacking_feat_imp
from ml.utils import dataset_root

df_mov = pd.read_csv(dataset_root + "/out/mov_res_folds.csv", sep="\t")
df_quest = pd.read_csv(dataset_root + "/out/quest_res_folds.csv", sep="\t")

with open(dataset_root + "/out/best_params_mov.json", "rb") as f:
    params_list_mov = json.load(f)
with open(dataset_root + "/out/best_params_quest.json", "rb") as f:
    params_list_quest = json.load(f)

with open("utils/info.json", "rb") as f:
    feature_importance_groups = json.load(f)
feature_importance_groups = {key: item for key, item in feature_importance_groups.items() if "group" in key}

all_res = {"imps": []}
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
        res, pred, imps, coefs = run_stacking_feat_imp(exp_list, params_list, feature_importance_groups, dataset_root, test_fold-1)
        res["pred"] = pred.tolist()
        all_res["imps"].append(imps[0])
        counter += 1

with open(f"{dataset_root}/out/stacking_imp_score.json", "w") as f:
    json.dump(all_res, f)

print("done")