import pandas as pd
import numpy as np
import utils.tools as tools 


'''
Question1: Use pandas to read data newAdult.csv. How many rows are there in this data set? 
            What are the column names? 
            Is there any personal Identifiable information in the data set?
            Is there any quasi-identifier information according to Sweeney's defination?
            Can you find possible quasi-identifier among the columns? Why are they?
            How unique the data is with age, fnlwgt, native_country being quasi-identifier?
'''

f = pd.read_csv('newAdult.csv')
# print(f.describe())
'''
# there are 32560 rows in the data set
'''
# print(f.columns)
# print(f.age.value_counts())
# print(f.fnlwgt.value_counts())
# print(f.native_country.value_counts())
f2 = f.drop_duplicates(['age','fnlwgt', 'native_country'])
# print(f2.describe()) # 29560 unique

'''
column = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
       'marital_status', 'moving', 'relationship', 'race', 'sex',
       'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
       'income']

there is no pii : https://piwik.pro/blog/what-is-pii-personal-data/
sex is quasi-identifier
age, fnlwgt, native_country. because value_counts have 1, so more likely to id people
'''

'''
Question 2: Suppose now you have a sensitive dataframe of 10,000 rows, link_attack.csv,
            which contains personal identifiable information, perform a link attack
            to re-identify people's name, ssn, race, and salary. You can use ['age',  'fnlwgt',
            'native_country']
    
'''
link_attack = pd.read_csv('link_attack.csv')
merged = pd.merge(f, link_attack, on = ['age','fnlwgt', 'native_country'])

# print(merged.head())
# print(merged.describe()) # 12144
# merged2 = merged.drop_duplicates(['age','fnlwgt', 'native_country'])
# print(merged2.head())
# print(merged2.describe()) # 9702
## 97% of them can be re-identified by link attack

'''
Question 3: Consider the definiaiton of k-anonymity, write a function to decide if 
            a data set is k-anonymous for a list of quasi-identifier
'''
qsi = ['age',  'fnlwgt',
        'sex',
       'native_country',
    ]
def is_k_anonymous(k, df, qsi):
    '''
    k: a constant to descide if a data set is k anonymous
    df: the data set of interest
    qsi: a list of quasi-identifier
    return: True if df is k-anonymous, False if not
    '''
    df_qsi = df.loc[:,qsi]
    # print(df_qsi)
    count = 0
    for index, row in df_qsi.iterrows():
        for index_c, compare_row in df_qsi.iterrows():
            if row.equals(compare_row):
                count += 1
                if count >= k:
                    break
        if count < k:
            return False
    return True

print(is_k_anonymous(2,f.head(2), qsi))
'''
this data set is not k anonymous with this quasi-identifers
'''