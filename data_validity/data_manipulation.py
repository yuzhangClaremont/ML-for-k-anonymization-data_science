import pandas as pd 
df = pd.read_csv('application_train.csv')
df.loc[0,'NAME_CONTRACT_TYPE'] = 'na'
df.loc[0:1,'SK_ID_CURR'] = 'unknown'
df.loc[[1,3,5],'CODE_GENDER'] = 'NA'
print(df.head())
df.to_csv('manipulated.csv')