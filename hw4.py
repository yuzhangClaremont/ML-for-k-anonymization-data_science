import pandas as pd
import numpy as np
import utils.tools as tools 


'''
Question1: Use pandas to read data newAdult.csv. How many rows are there in this data set? 
            What are the column names? 
            Is there any personal Identifiable information in the data set?
            Is there any quasi-identifier information according to Sweeney's defination?
            Can you find possible quasi-identifier among the columns? Why are they?
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
Question 2: Consider the definiaiton of k-anonymity, write a function to decide if 
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