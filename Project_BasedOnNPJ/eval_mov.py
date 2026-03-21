import pandas as pd
from ml.utils import dataset_root
import json
import numpy as np

experiments = pd.read_csv("utils/experiments.csv")
experiments = experiments[experiments["data"] == "movement"]

params_list = []
df_list = []
for row_idx, exp in experiments.iterrows():
    with open(f"{dataset_root}/out/hyperopt/{exp['exp_name']}.json", "r") as f:
        exp_res = json.load(f)
    new_res = {}
    for key, item in exp_res["test_score"].items():
        new_res["test_" + key] = np.round(np.array(item) * 100, decimals=2)
    new_res["hyperopt_balanced_accuracy"] = exp_res["score"]
    new_params = []
    for params in exp_res["params"]:
        params_list.append(params)
        params_str = []
        for key, item in params.items():
            params_str.append(f"{key}: {item}")
        params_str = ", ".join(params_str)
        new_params.append(params_str)
    new_res["params"] = params_str
    new_res["mode"] = [exp["mode"] for n in range(5)]
    new_res["data"] = [exp["data"] for n in range(5)]
    new_res["experiment"] = [exp["experiment"] for n in range(5)]
    new_res["channel_filter"] = [exp["channel_filter"] for n in range(5)]
    new_res["classifier"] = [exp["classifier"] for n in range(5)]
    new_res["test_fold"] = [n for n in range(1, 6)]
    new_res["exp_name"] = [exp["exp_name"] for n in range(5)]
    exp_df = pd.DataFrame(new_res)
    df_list.append(exp_df)
df = pd.concat(df_list, ignore_index=True, axis=0)
df_grouped = df.groupby(["mode", "test_fold"])
idx_max = df_grouped["hyperopt_balanced_accuracy"].idxmax()

df_folds = df.iloc[idx_max, :]
df_folds.to_csv(dataset_root + "/out/mov_res_folds.csv", index=False, sep="\t")

params_folds = [params_list[idx] for idx in idx_max]
with open(dataset_root + "/out/best_params_mov.json", "w") as f:
    json.dump(params_folds, f, indent=2)

df_grouped = df_folds.groupby(["mode"])
df_mean = df_grouped[["test_accuracy", "test_balanced_accuracy", "test_f1", "test_precision", "test_recall", "test_roc_auc"]].mean()
df_std = df_grouped[["test_accuracy", "test_balanced_accuracy", "test_f1", "test_precision", "test_recall", "test_roc_auc"]].std()
df_mean = df_mean.apply(lambda col: col.round(2).astype(str)) \
          + " (" + df_std.apply(lambda col: col.round(2).astype(str)) + ")"
df_mean.to_csv(dataset_root + "/out/mov_res.csv", index=False, sep="\t")
