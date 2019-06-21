""" 
Author: Yun Zhang
Class: CS 181R
Week 7 - Data Privacy and Anonymity
Homework 4
Name:
"""
import pandas as pd
import numpy as np
import os

df = pd.read_csv('data/adult.data')

'''
Question1:  Use pandas to explore the data set from 1994 Census database. 
            How many rows are there in this data set?  
            Add column names to this data set. The list of column names are: ['age', 'workclass', 
            'fnlwgt', 'education', 'education_num', 'marital_status', 'moving', 'relationship', 
            'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country','income']
            Is there any personal Identifiable information in the data set?
            Is there any quasi-identifier information according to Sweeney's defination?
            Can you find possible quasi-identifier among the columns? Why are they?

            Write some codes after TODO. Then write your response in a seperate file and save it as hw4.pdf.
'''

# TODO: Add column names to df. How many rows are there in this data set? 
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'moving', 'relationship', 'race', 'sex',
'capital_gain','capital_loss', 'hours_per_week', 'native_country','income']

print('there are', df.shape[0], 'rows in df')

'''
there are 32560 rows in the data set
'''

# TODO      : Is there any personal Identifiable information in the data set?
#           Is there any quasi-identifier information according to Sweeney's defination?
#           Can you find possible quasi-identifier among the columns? 
#           Why do you think they can be quasi-identifier?
#           Hint: use value_counts() to observe how unique a values is for each column. Column with more unique
#               values are more likely to be used as quasi-identifier.


# print(f.age.value_counts())
# print(f.fnlwgt.value_counts())
# print(f.native_country.value_counts())
'''
there is no pii : https://piwik.pro/blog/what-is-pii-personal-data/
sex is quasi-identifier in Sweeny's defination
age, fnlwgt, native_country. because value_counts have 1, so more likely to id people
'''

'''
Question 2: Your supervisor believes the information in ['fnlwgt', 'education', 'relationship', 
            'capital_gain', 'capital_loss', 'hours_per_week'] are not necessary for your analysis.
            Use suppression technique to de-idenitify this data set. Name the new data frame 
            variable 'suppressed' and save it as 'suppressed.csv' without a header, without index. 
            How many rows are unique in the 'suppressed' data set? 
'''
# TODO: Use suppression technique to de-idenitify this data set. 
#       Then save the new data set under data directory as 'suppressed.csv' without a header, without index.

suppressed = df.drop(columns = ['fnlwgt', 'education', 'relationship', 
                            'capital_gain', 'capital_loss', 'hours_per_week'])

suppressed.to_csv('data/suppressed.csv', header = False, index = False)


# TODO: How many rows in suppressed data set are unique?
unique = suppressed.drop_duplicates(keep = False)
# print(unique.head()) 
# print(unique.describe()) 
'''
17047 unique
'''


'''
Question 3: Suppose now you have a sensitive dataframe of 10,000 rows named as "link_attack.csv",
            which contains personal identifiable information. Take a look at the head of this attack data,
            perform a link attack and to re-identify people's name, ssn, race, and salary. You can use 
            ['age',  'sex, 'native_country'] as quasi-identifier. How many people can you identify?
 
'''
# TODO: your code here
link_attack = pd.read_csv('data/link_attack.csv')
merged = pd.merge(suppressed, link_attack, on = ['age','sex', 'native_country'])

merged_unique = merged.drop_duplicates(['age','sex', 'native_country'], keep = False)
# print(merged_unique.head())
# print(merged_unique.describe()) 
'''
287 people can be identified by link attack
'''


'''
Question 4: Consider the definiaiton of k-anonymity, write a function to decide if 
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
# print(is_k_anonymous(2, suppressed, qsi))

'''
this data set is not 2 anonymous with this quasi-identifers
'''

'''
Question 5: Now we wish to generalize the informations in adult.csv to make it k-anonymous while minimising
            the lose of information as little as possible. This problem turns out to be NP-Hard, which means 
            there is no polynomial algorithm to solve it. However, a compariably fast algorithm can solve it
            sub-optimal. The detail of this algorithm can be found at: https://github.com/qiyuangong/Mondrian
            Read through it, and use this package to make this data set 2-anonymous.
'''
cmd = 'python3 anonymizer.py s a 2'

# TODO: Use os module to run cmd command. 
#       then add column names 
#       back to the anonymized data set. Observe the data by printing out the head and describe funtion.

os.system(cmd) 

# TODO: Read the anonymized data generated by this Mondrian implementation and save it in anony variable.
#        Then add column names ['age', 'workclass',  'education_num','marital_status', 'moving', 'race', 
#       'sex','native_country','income'] and observe this data set.
anony = pd.read_csv('data/anonymized.csv')
anony.columns = ['age', 'workclass',  'education_num',
       'marital_status', 'moving', 'race', 'sex',
        'native_country','income']
# print('k-anonymized data !!!!')
# print(anony.head())
# print(anony.describe())


# TODO: Test if it is k-anonymous using the function in Question 3 and same quasi-identifier. 
#       How many rows in this data set are unique now? We can call the non-duplicate data set
#       as unique anony.

# print(is_k_anonymous(2,anony, qsi))

unique_anony = anony.drop_duplicates(['age', 'sex', 'native_country'], keep = False)
# print(unique_anony.head()) #
# print(unique_anony.describe()) #

'''
It is 2-anonymous now. And there are 0 unique rows with qsi = ['age', 'sex', 'native_country']
'''

# monAnony = pd.read_csv('data/monAdult.csv')
# print(monAnony.describe())
'''
Question 6: Use the anonymized data set, analyze what percentage of male and female has income 
            less than 50k? Compare the result with the original datase 'adult.data'.
'''
sex_group = anony.groupby(['sex'])
for sex, sex_df in sex_group:
    print(sex)
    print(sex_df.describe())

# male 13685/20017, female 8002/9062

print('original data!!!!!!!')
sex_group2 = df.groupby(['sex'])
for sex, sex_df in sex_group:
    print(sex)
    print(sex_df.describe())

# male 13685/20017, female 8002/9062

def main():
    # test 1
    assert np.array_equal(df.columns, ['age', 'workclass', 
            'fnlwgt', 'education', 'education_num', 'marital_status', 'moving', 'relationship', 
            'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country','income']) == True
    # test 2
    assert np.array_equal(suppressed.columns, ['age', 'workclass', 'education_num',
            'marital_status', 'moving', 'race', 'sex', 'native_country','income'] ) == True

    assert unique.shape[0] == 17047

    # test 3
    assert merged_unique.shape[0] == 287

    # test 4
    assert is_k_anonymous(2, suppressed, qsi) == False

    two_anon = pd.DataFrame(
    [
        ['*', 'Storm',34,'Black'],
        ['John','*','*','*'],
        ['*', 'Storm',34,'Black'],
        ['John','*','*','*']
    ],
    columns = ['First','Last','age','race']
)
    assert is_k_anonymous(2, two_anon, ['First','Last','age','race']) == True

    # test 5
    assert is_k_anonymous(2,anony, qsi) == True
    assert unique_anony.shape[0] == 0

    # test 6
    # print(df.groupby('sex').get_group(' Male')['income'].value_counts().tolist())
    # print(df.groupby('sex').groups.keys())
    # assert anony.groupby('sex').get_group('Male')['income'].value_counts().tolist()[0] == 13685
    # assert df.groupby('sex').get_group(' Male')['income'].value_counts().tolist()[0] == 13685



if __name__ == "__main__":
    main()



