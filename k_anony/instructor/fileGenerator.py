import pandas as pd
import numpy as np
import os

# add columns and write adult.csv
FILE = os.path.join('data', 'adult.data')
df = pd.read_csv(FILE)
print(df.head())
print(df.shape)
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain','capital_loss', 'hours_per_week', 'native_country','income']
PATH = os.path.join('data', 'adult.csv')

print(df.head())
print(df.shape)
df.to_csv(PATH, index = False)

ATTACK = os.path.join('data', 'link_attack.csv')
attack = pd.read_csv(ATTACK)
tinyAttack = attack.iloc[2000:2100,:]
print(tinyAttack)
PATH = os.path.join('data', 'tinyAttack.csv')
tinyAttack.to_csv(PATH, index = False)