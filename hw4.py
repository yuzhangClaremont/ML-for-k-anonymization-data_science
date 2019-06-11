import pandas as pd
import numpy as np
import utils.tools as tools 


'''
Question1: Use pandas to read data newAdult.csv, a data set from 1994 Census database (https://archive.ics.uci.edu/ml/datasets/adult).
            How many rows are there in this data set? 
            What are the column names? 
            Is there any personal Identifiable information in the data set?
            Is there any quasi-identifier information according to Sweeney's defination?
            Can you find possible quasi-identifier among the columns? Why are they?
            How many unique the data can be identified with age, fnlwgt, native_country being quasi-identifier?
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
Question 2: Suppose now you have a sensitive dataframe of 10,000 rows named as "link_attack.csv",
            which contains personal identifiable information. Perform a link attack
            to re-identify people's name, ssn, race, and salary. You can use ['age',  'fnlwgt',
            'native_country'] as quasi-identifier. How many people can you identify?
    
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
            a data set is k-anonymous for a list of quasi-identifier. Then test if 
            this data set adult.csv is 2 anonymous with ['age', 'sex','native_country']
            being quasi-identifiers
'''

def is_k_anonymous(k, df, qsi):
    '''
    k: a constant to descide if a data set is k anonymous
    df: the data set of interest
    qsi: a list of quasi-identifier
    return: True if df is k-anonymous, False if not
    '''
    df_qsi = df.loc[:3,qsi]
    # print(df_qsi)
    count = 0
    for index, row in df_qsi.iterrows():
        for index_c, compare_row in df_qsi.iterrows():
            # print(compare_row)
            if row.equals(compare_row):
                count += 1
                if count >= k:
                    break
                
        if count < k:
            return False
    return True

def is_k_anonymous2(k, df, qsi):
    '''
    k: a constant to descide if a data set is k anonymous
    df: the data set of interest
    qsi: a list of quasi-identifier
    return: True if df is k-anonymous, False if not
    '''
    df_qsi = df.loc[:,qsi]
    # print(df_qsi)
    sythesized = df_qsi[qsi[0]].map(str) # + anon['Zip'].map(str)
    for i in range(1,len(qsi)):
        sythesized += df_qsi[qsi[i]].map(str) 
    # sythesized.map(str)
    # dup = df_qsi[df_qsi.duplicated(keep = False)] 
    count = sythesized.value_counts(ascending = True)
    if count[0] < k:
        return False
    else:
        return True
    # print(count[0] )

qsi = ['age', 'sex', 'native_country']
# print(is_k_anonymous2(2,f, qsi))

'''
this data set is not 2 anonymous with this quasi-identifers
'''

'''
Question 4: Now we wish to generalize the informations in adult.csv to make it k-anonymous while minimising
            the lose of information as little as possible. This problem turns out to be NP-Hard, which means 
            there is no polynomial algorithm to solve it. However, a compariably fast algorithm can solve it
            sub-optimal. The detail of this algorithm can be found at: https://github.com/qiyuangong/Mondrian
            Read through it, and use this package to make this data set 2-anonymous.
'''
import os
cmd = 'python anonymizer.py s "adult.csv" 2'
# anony = pd.read_csv()