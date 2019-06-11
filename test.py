import pandas as pd
import numpy as np
attack = pd.read_csv('adult_with_pii.csv')
attack = attack.drop(0)
print(attack.describe())
# print(attack.describe() )
adult = pd.read_csv('newAdult.csv')
print(adult.describe() )
name = attack.Name 
ssn = attack.SSN 
age = adult.age
fnlwgt = adult.fnlwgt
native_country = adult.native_country

res = pd.concat([name,ssn, age, fnlwgt, native_country], axis = 1)
link_attack = res[10000:20000]
# print(link_attack.head())
# print(link_attack.describe())
link_attack.to_csv('link_attack.csv', index = False)

repeats = adult[adult.age == 25 ] # and adult.fnlwgt== 176756 and adult.native_country == 'United-States'
repeats = repeats[repeats.fnlwgt== 176756 ]
# repeats = repeats[repeats.native_country.str.contains('United-States') ]
repeats = repeats[repeats.native_country.str.match'United-States') ]
print(repeats) # two rows, not identifiable