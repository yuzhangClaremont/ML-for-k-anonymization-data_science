import pandas as pd
import numpy as np
# import utils.tools as tools 


'''
Question1:  Use pandas to read data newAdult.csv, a data set from 1994 Census database (retrived
            from: https://archive.ics.uci.edu/ml/datasets/adult).
            How many rows are there in this data set? 
            What are the column names? 
            Is there any personal Identifiable information in the data set?
            Is there any quasi-identifier information according to Sweeney's defination?
            Can you find possible quasi-identifier among the columns? Why are they?
            How many unique the data can be identified with age, fnlwgt, native_country being quasi-identifier?
'''

f = pd.read_csv('newAdult.csv')
# TODO: How many rows are there in this data set? What are the column names? 

# print(f.describe())
# print(f.columns)
'''
there are 32560 rows in the data set
column = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
       'marital_status', 'moving', 'relationship', 'race', 'sex',
       'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
       'income']
'''

# TODO: Is there any personal Identifiable information in the data set?
#       Is there any quasi-identifier information according to Sweeney's defination?
#       Can you find possible quasi-identifier among the columns? Why are they?

# print(f.age.value_counts())
# print(f.fnlwgt.value_counts())
# print(f.native_country.value_counts())
'''
there is no pii : https://piwik.pro/blog/what-is-pii-personal-data/
sex is quasi-identifier in Sweeny's defination
age, fnlwgt, native_country. because value_counts have 1, so more likely to id people
'''

'''
Question 2: Your supervisor believes the information in ['fnlwgt', 'education', 'relationship', 'capital_gain', 'capital_loss', 'hours_per_week']
            are not necessary for your analysis. Use suppression technique to de-idenitify this data set.
'''
# TODO: Use suppression technique to de-idenitify this data set. Then save the new data set under data directory as 'suppressed.csv' without a header.

dropped = f.drop(columns = ['fnlwgt', 'education', 'relationship', 'capital_gain', 'capital_loss', 'hours_per_week'])
print(dropped.head())
dropped.to_csv('data/suppressed.csv', header = False, index = False)


# print(f.age.value_counts())
# print(f.fnlwgt.value_counts())
# print(f.native_country.value_counts())

f2 = f.drop_duplicates(['age','sex', 'native_country'])
print(f2.head()) 
print(f2.describe()) # 29560 unique



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

# f.columns = [''] * len(f.columns)
f.to_csv(r'data/newAdult4.csv',index=False, header = False)

# TODO: Use os module to run command 'python3 anonymizer.py s a 2'. Read the generated data 'data/anonymized.csv'
import os
cmd = 'python3 anonymizer.py s a 2'
os.system(cmd) 
anony = pd.read_csv('data/anonymized.csv')

# TODO: Add columns as ['age', 'workclass',  'education_num','marital_status', 'moving', 'race', 'sex','native_country','income']
anony.columns = ['age', 'workclass',  'education_num',
       'marital_status', 'moving', 'race', 'sex',
        'native_country','income']

# TODO: Print the head of the anonymized data, then test if it is k-anonymous using the function in Question 3
# print(anony.head())
# print(is_k_anonymous(2,anony, qsi))