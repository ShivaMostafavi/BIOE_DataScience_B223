from ml.utils import get_file_list
import pandas as pd


df = get_file_list()

df.drop(columns=['resource_type', 'id', 'study_id', 'disease_comment',
                 'handedness', 'appearance_in_kinship', 'appearance_in_first_grade_kinship',
                 'effect_of_alcohol_on_tremor', 'label'], inplace=True)

# Print sample size
sample_size = df.groupby(by=['condition']).size()
print('Sample size:')
print(sample_size)
print('Sample size by gender:')
# Print sample size by gender
sample_size_gender = df.groupby(by=['condition', 'gender']).size()
print(sample_size_gender)

df.drop(columns=['gender'], inplace=True)
# Print mean basic statistics
df['age_at_diagnosis'] = pd.to_numeric(df['age_at_diagnosis'], errors='coerce')

mean_vals = df.groupby(by=['condition']).mean().round(decimals=2)
std_vals = df.groupby(by=['condition']).std().round(decimals=2)
print(mean_vals)
print(std_vals)
