import json
import pandas as pd
from ml.utils import dataset_root
from utils.experiment import experiment

df_quest = pd.read_csv(dataset_root + "/out/quest_res_folds.csv", sep="\t")

with open(dataset_root + "/out/best_params_quest.json", "rb") as f:
    params_list_quest = json.load(f)

all_res = []
counter = 0
for mode in ["pd_vs_dd", "pd_vs_hc"]:
    for test_fold in range(1, 6):
        exp = df_quest[(df_quest["mode"] == mode) & (df_quest["test_fold"] == test_fold)]
        exp = exp.iloc[0, :]
        params = {exp["classifier"]: params_list_quest[counter]}

        res, pred = experiment(exp, params, dataset_root, test_fold-1)
        res["pred"] = pred.tolist()
        res["mode"] = mode
        all_res.append(res)
        counter += 1

df = pd.DataFrame(all_res)
df.loc[:, ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]] = df.loc[:, ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]].map(lambda x: x[0])
# df.to_csv(f"{dataset_root}out/quest_res_folds.csv", index=False, sep="\t")

df_grouped = df.groupby(["mode"])
df_mean = df_grouped[["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]].mean().astype(float)
df_std = df_grouped[["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"]].std().astype(float)
df_mean = df_mean.apply(lambda col: (col * 100).round(2).astype(str)) + " (" + df_std.apply(lambda col: (col * 100).round(2).astype(str)) + ")"
df_mean.to_csv(f"{dataset_root}/out/quest_res.csv", index=False, sep="\t")
print("done")