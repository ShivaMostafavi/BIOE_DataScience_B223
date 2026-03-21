import json
from ml.utils import dataset_root
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open("utils/info.json", "rb") as f:
    feature_importance_groups = json.load(f)

groups = [group for key, item in feature_importance_groups.items() if "group" in key for group in item]

with open(dataset_root + "out/stacking_imp_score.json", "rb") as f:
    res_dict = json.load(f)

all_imps = np.array(res_dict["imps"])

# PD vs. DD
imps = all_imps[:5]
imps = np.mean(imps, axis=2)
imps = np.mean(imps, axis=0)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), sharey="all")

df = pd.Series(index=groups, data=imps)
df.sort_values(ascending=False, inplace=True)
df = df.head(8)

df.plot.bar(ax=ax[1], legend=False, color="#005628")
ax[1].set_ylabel("Information gain", fontsize=12)
ax[1].set_title("PD vs. DD", fontsize=14)

# PD vs. HC
imps = all_imps[5:]
imps = np.mean(imps, axis=2)
imps = np.mean(imps, axis=0)

df = pd.Series(index=groups, data=imps)
df.sort_values(ascending=False, inplace=True)
df = df.head(8)

df.plot.bar(ax=ax[0], legend=False, color="#005628")
ax[0].set_title("PD vs. HC", fontsize=14)


plt.tight_layout()
plt.savefig(dataset_root + "imgs/" + "feature_importance.svg")
plt.close()

print("done")