import pandas as pd
import numpy as np
attack = pd.read_csv('adult_with_pii.csv')
attack = attack.drop(1)
print(attack.head())
# print(attack.describe() )
adult = pd.read_csv('newAdult.csv')
print(adult.head() )
name = attack.Name 
ssn = attack.SSN 
age = adult.age
fnlwgt = adult.fnlwgt
native_country = adult.native_country

res = pd.concat([name,ssn, age, fnlwgt, native_country], axis = 1)
print(res.head())
print(res.describe())