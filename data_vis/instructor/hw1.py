import numpy as np
import pandas as pd

# The following two modules matplotlib and seaborn are for plots
import matplotlib.pyplot as plt
import seaborn as sns # Comment this if seaborn is not installed
# %matplotlib inline

# The module re is for regular expressions
import re
import os

PATH = os.path.join('data','train.csv')
df = pd.read_csv(PATH)
# print(df.describe())

# age_group = {'0-9':0, '10-19':0,'20-29':0, '30-39':0,'40-49':0, 
#             '50-59':0,'60-69':0, '70-79':0, '80-90':0}
# for index, row in df.iterrows():
#     if row['Age'] < 10:
#         age_group['0-9'] += 1

df['age_group'] = pd.cut(df['Age'],[0,10,20,30,40,50,60,70,80]).astype(str)
age_groups = df.groupby('age_group').sum()
print(age_groups)
age_group_count = df['age_group'].value_counts()
print(age_group_count)
survive_rate = age_groups['Survived'] / age_group_count
print(survive_rate)
sns.barplot(x= survive_rate.index, y = survive_rate.values)
plt.show(sns)

