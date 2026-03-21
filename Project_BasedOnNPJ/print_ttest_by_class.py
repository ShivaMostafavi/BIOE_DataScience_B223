import itertools
from scipy.stats import chi2_contingency, ttest_ind
import numpy as np
from ml.utils import get_file_list


df = get_file_list()

df.drop(columns=['resource_type', 'id', 'study_id', 'disease_comment',
                 'handedness', 'appearance_in_kinship', 'appearance_in_first_grade_kinship',
                 'effect_of_alcohol_on_tremor', 'condition', 'age_at_diagnosis'], inplace=True)

df = df.groupby(by=['label'])
groups = list(df.groups.keys())

# Loop
print('T-test: age')
for comb in list(itertools.combinations(groups, 2)):
    vals1 = df.get_group(comb[0])['age'].values
    vals2 = df.get_group(comb[1])['age'].values
    stat, pvalue = ttest_ind(vals1, vals2)
    print(f'{comb}: {pvalue:.5f}, <0.01: {pvalue < 0.01}')

print('Chi-square: gender')
for comb in list(itertools.combinations(groups, 2)):
    vals1 = df.get_group(comb[0]).groupby('gender').size().values
    vals2 = df.get_group(comb[1]).groupby('gender').size().values
    vals = np.stack([vals1, vals2])
    _, pvalue, _, _ = chi2_contingency(vals)
    print(f'{comb}: {pvalue:.5f}, <0.01: {pvalue < 0.01}')
