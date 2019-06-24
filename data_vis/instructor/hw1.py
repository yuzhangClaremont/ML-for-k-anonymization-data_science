import numpy as np
import pandas as pd

# The following two modules matplotlib and seaborn are for plots
import matplotlib.pyplot as plt
import seaborn as sns # Comment this if seaborn is not installed
# %matplotlib inline

# The module re is for regular expressions
import re
import os

# Question 1
PATH = os.path.join('data','train.csv')
df = pd.read_csv(PATH)
print(df.dtypes)
for c in df.columns:
    print(c, df[c].isnull().sum())

df.drop(columns = ['Cabin'], inplace = True)
print(df.isnull().sum())


'''
1a. catigorical and continueous should be classified differently. float, int or object
1.b. pclass, sex, age, cabin
no data can be used for other purposes
1c. 891
1d. age and cabin

'''
# Q2
def survived(df):
    res = df[ df['Survived'] == 1]
    res = df.loc[ df['Survived'] == 1]
    return res
survived_df = survived(df)
print(survived_df.describe())

def survive_rate(df, feature):
    base = df[feature].value_counts()
    survived_df = survived(df)
    anominator = survived_df[feature].value_counts()
    return anominator/base
print('!!!',survive_rate(df, 'Parch'))

'''
!!! female    0.742038
male      0.188908

!!! 1    0.629630
2    0.472826
3    0.242363

!!! 0    0.343658
1    0.550847
2    0.500000
3    0.600000
4         NaN
5    0.200000
6         NaN

'''

def survived_median_age(df):
    # survived_df = survived(df)
    df.dropna(subset = ['Age'], inplace = True)
    group_median = df.groupby('Survived').median()
    return group_median['Age']

print(survived_median_age(df))

def survived_mean_age(df):
    # survived_df = survived(df)
    df.dropna(subset = ['Age'], inplace = True)
    group_mean = df.groupby('Survived').mean()
    return group_mean['Age']
print('mean')
print(survived_mean_age(df))

print(df['Survived'].value_counts())

# age_group = {'0-9':0, '10-19':0,'20-29':0, '30-39':0,'40-49':0, 
#             '50-59':0,'60-69':0, '70-79':0, '80-90':0}
# for index, row in df.iterrows():
#     if row['Age'] < 10:
#         age_group['0-9'] += 1

df['age_group'] = pd.cut(df['Age'],[0,10,20,30,40,50,60,70,80]).astype(str)
age_groups = df.groupby('age_group').sum()
# print(age_groups)
age_group_count = df['age_group'].value_counts()
# print(age_group_count)
survive_rate = age_groups['Survived'] / age_group_count
# print(survive_rate)
# sns.barplot(x= survive_rate.index, y = survive_rate.values)
# sns.pointplot(x='Sex', y='Survived', hue='Pclass', data=df)
# plt.show(sns)

# df.dropna(axis = 1,  inplace = True)
# print(re.findall("\w+[.]", 'Heikkinen, Miss. Laina'))
# print(df['Age'].astype(str)[0][0])

# 5.b
df.dropna(subset = ['Age'], inplace = True)
# print( df )
survived_median = df.groupby('Survived').median()
# dfS.plot(kind='pie', y = 'Age',autopct='%1.1f%%')
# df['MedianAge'] = df.groupby('Survived')['Age'].transform("median")
# df.loc[0:100,['MedianAge']] = 100
survived_median.rename(columns = {'Age': 'MedianAge'}, inplace = True)
# print(dfS.head())
sns.barplot(x = survived_median.index, y = survived_median.Age)
plt.show(sns)
# print(dfS)