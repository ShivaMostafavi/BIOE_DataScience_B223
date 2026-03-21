from pathlib import Path
import pandas as pd
from utils.hyperopt import hyperopt
from ml.utils import dataset_root
import json

Path(dataset_root + '/out/hyperopt').mkdir(parents=True, exist_ok=True)
Path(dataset_root + '/out/boss').mkdir(parents=True, exist_ok=True)

experiments = pd.read_csv('utils/experiments.csv')
with open('utils/hyperparams.json', 'rb') as f:
    hyperparams = json.load(f)

for row_idx, exp in experiments.iterrows():
    print(f'Run {row_idx + 1} / {len(experiments)}: {exp["exp_name"]}')
    res = hyperopt(exp, hyperparams, dataset_root)
    with open(f'{dataset_root}/out/hyperopt/{exp["exp_name"]}.json', 'w') as f:
        json.dump(res, f)
