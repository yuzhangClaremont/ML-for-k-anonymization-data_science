import pandas as pd 

missing_values = ["n/a", "na", "--", "unknown"]
df = pd.read_csv('manipulated.csv', na_values = missing_values)

# df = pd.read_csv('manipulated.csv')

# print(df.describe()) # 307511?
print(df.isnull().sum())