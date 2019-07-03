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

'''
Question 2: Your supervisor believes the information in ['fnlwgt', 'education', 'relationship', 
            'capital_gain', 'capital_loss', 'hours_per_week'] are not necessary for your analysis.
            Use suppression technique to de-idenitify this data set. Name the new data frame 
            variable 'suppressed' and save it as 'suppressed.csv' without a header, without index. 
            How many rows are unique in the 'suppressed' data set? 
'''
# TODO: Write a function suppression() which input a DataFrame df, and a list of column names 
#       to drop these columns from df. Then save the new data set under data directory as 
#       'suppressed.csv' without a header, without index.

def suppression(df, column_names):
    """
    input: df, a DataFrame to suppress. column_names: a list of strings
    output: the path where 'suppressed.csv' is saved
    """
    suppressed = df.drop(columns = column_names)
    path = os.path.join('data', 'suppressed.csv')
    suppressed.to_csv(path, header = False, index = False)
    return path

'''
Question 3: Suppose now you have a sensitive dataframe of 10,000 rows named as "link_attack.csv",
            which contains personal identifiable information. Take a look at the head of this attack data,
            perform a link attack and to re-identify people's name, ssn, race, and salary. You can use 
            ['age',  'sex, 'native_country'] as quasi-identifier. How many people can you identify?
 
'''

# TODO: Implement function link_attack(df, attack_df, qsi)
def link_attack(df, attack_df, qsi):
    """
    input:
    output:
    """
    print(qsi)
    merged = pd.merge(df, attack_df, on = qsi)
    return merged


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

# qsi = ['age', 'sex', 'native_country']
# print(is_k_anonymous(2, suppressed, qsi))

'''
this data set is not 2 anonymous with this quasi-identifers
'''



# monAnony = pd.read_csv('data/monAdult.csv')
# print(monAnony.describe())
'''
Question 6: Use the anonymized data set, analyze what percentage of male and female has income 
#             less than 50k? Compare the result with the original datase 'adult.data'.
# '''
# sex_group = anony.groupby(['sex'])
# for sex, sex_df in sex_group:
#     print(sex)
#     print(sex_df.describe())

# # male 13685/20017, female 8002/9062

# print('original data!!!!!!!')
# sex_group2 = df.groupby(['sex'])
# for sex, sex_df in sex_group:
#     print(sex)
#     print(sex_df.describe())

# male 13685/20017, female 8002/9062

def main():
    PATH = os.path.join('data', 'adult.data')
    df = pd.read_csv(PATH)

    '''
    Question1:  In the main function of hw4.py, use pandas functions to print out DataFrame information 
                and answer questions in your writeup pdf file.

                Is there any personal Identifiable information in the data set?
                Is there any quasi-identifier information according to Sweeney's defination?
                Can you find possible quasi-identifier among the columns? Why are they?

                Write some codes after TODO. Then write your response in a seperate file and save it as hw4.pdf.
    '''

    # TODO:      Add column names to this data set. The list of column names are: ['age', 'workclass', 
    #           'fnlwgt', 'education', 'education_num', 'marital_status', 'moving', 'relationship', 
    #            # 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country','income']
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'moving', 'relationship', 'race', 'sex',
    'capital_gain','capital_loss', 'hours_per_week', 'native_country','income']

    # TODO:      How many rows are there in this data set?
    print('there are', df.shape[0], 'rows in df~~~')

    '''
    there are 32560 rows in the data set
    '''

    # TODO      Write some code to explore the DataFrame.
    #           Can you find possible quasi-identifier among the columns? 
    #           Why do you think they can be quasi-identifier?


    # print(f.age.value_counts())
    # print(f.fnlwgt.value_counts())
    # print(f.native_country.value_counts())
    '''
    there is no pii : https://piwik.pro/blog/what-is-pii-personal-data/
    sex is quasi-identifier in Sweeny's defination
    age, fnlwgt, native_country. because value_counts have 1, so more likely to id people
    '''

    """
    Question 2
    # TODO: How many rows in suppressed data set are unique?
    """
    drop_column_names = ['fnlwgt', 'education', 'relationship', 'capital_gain', 'capital_loss', 'hours_per_week']
    suppressed_path = suppression(df, drop_column_names)
    suppressed_df = pd.read_csv(suppressed_path)
    suppressed_df.columns = ['age', 'workclass',  'education_num',
       'marital_status', 'moving', 'race', 'sex',
        'native_country','income']

    unique = suppressed_df.drop_duplicates(keep = False)
    # print(unique.head()) 
    # print(unique.describe()) 
    '''
    17048 unique
    '''

    """
    Question 3
    """
    ATTACK_PATH = os.path.join('data', 'link_attack.csv')
    link_attack_df = pd.read_csv(ATTACK_PATH)
    qsi = ['age','sex', 'native_country']
    # TODO: Run link_attack you implemented above in the main function with the suppressed 
    #       DataFrame, the attack DataFrame. Write code to find how many people can you re-identify? 
    #       You can use ['age',  'sex, 'native_country'] as the quasi-identifiers.
    merged_df = link_attack(suppressed_df, link_attack_df,  qsi )

    merged_unique = merged_df.drop_duplicates(['age','sex', 'native_country'], keep = False)
    # print(merged_unique.head())
    # print(merged_unique.describe()) 
    '''
    287 people can be identified by link attack
    '''

    """
    Question 4: Use is_k_anonymous(k, df, qsi)to test if the suppressed data set is 2 anonymous.
                 Use ['age', 'sex','native_country'] as quasi-identifiers.
    """
    print('is suppressed file k anonymous???? ',is_k_anonymous(2,suppressed_df, qsi))

    '''
    Question 5: Now we wish to generalize the informations in adult.csv to make it k-anonymous while minimising
                the lose of information as little as possible. This problem turns out to be NP-Hard, which means 
                there is no polynomial algorithm to solve it. However, a compariably fast algorithm can solve it
                sub-optimal. The detail of this algorithm can be found at: https://github.com/qiyuangong/Mondrian
                Read through it, and use this package to make this data set 2-anonymous.
    '''
    cmd = 'python3 anonymizer.py s a 2'

    # TODO: Use os module to run cmd command. 
    #       Read the anonymized data generated by this Mondrian implementation and save it in anony variable.
    #        Then add column names ['age', 'workclass',  'education_num','marital_status', 'moving', 'race', 
    #       'sex','native_country','income'] and observe this data set.
    #       Test if it is k-anonymous using the function in Question 3 and same quasi-identifier.

    os.system(cmd) 
    anony = pd.read_csv('data/anonymized.csv')
    anony.columns = ['age', 'workclass',  'education_num',
        'marital_status', 'moving', 'race', 'sex',
            'native_country','income']
    # print('k-anonymized data !!!!')
    # print(anony.head())
    # print(anony.describe())
    print('is anony file k anonymous? ',is_k_anonymous(2,anony, qsi))


    # TODO: How many rows in this data set are unique now? We can call the non-duplicate data set
    #       as unique anony.

    unique_anony = anony.drop_duplicates(['age', 'sex', 'native_country'], keep = False)
    # print(unique_anony.head()) #
    # print(unique_anony.describe()) #

    '''
    It is 2-anonymous now. And there are 0 unique rows with qsi = ['age', 'sex', 'native_country']
    '''

#     # test 1
#     assert np.array_equal(df.columns, ['age', 'workclass', 
#             'fnlwgt', 'education', 'education_num', 'marital_status', 'moving', 'relationship', 
#             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country','income']) == True
#     # test 2
#     assert np.array_equal(suppressed.columns, ['age', 'workclass', 'education_num',
#             'marital_status', 'moving', 'race', 'sex', 'native_country','income'] ) == True

#     assert unique.shape[0] == 17047

#     # test 3
#     assert merged_unique.shape[0] == 287

#     # test 4
#     assert is_k_anonymous(2, suppressed, qsi) == False

#     two_anon = pd.DataFrame(
#     [
#         ['*', 'Storm',34,'Black'],
#         ['John','*','*','*'],
#         ['*', 'Storm',34,'Black'],
#         ['John','*','*','*']
#     ],
#     columns = ['First','Last','age','race']
# )
#     assert is_k_anonymous(2, two_anon, ['First','Last','age','race']) == True

#     # test 5
#     assert is_k_anonymous(2,anony, qsi) == True
#     assert unique_anony.shape[0] == 0

    # test 6
    # print(df.groupby('sex').get_group(' Male')['income'].value_counts().tolist())
    # print(df.groupby('sex').groups.keys())
    # assert anony.groupby('sex').get_group('Male')['income'].value_counts().tolist()[0] == 13685
    # assert df.groupby('sex').get_group(' Male')['income'].value_counts().tolist()[0] == 13685



if __name__ == "__main__":
    main()
    print("hello!!!")



