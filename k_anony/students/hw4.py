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
Question 1 code in main funciton
'''


'''
Question 2: Your supervisor believes the information in ['fnlwgt', 'education', 'relationship', 
            'capital_gain', 'capital_loss', 'hours_per_week'] are not necessary for your analysis.
            Write a function suppression() to de-idenitify this df with suppression technique.
            Save the suppressed data set as 'suppressed.csv' in "data" directory without header and index . 
            The function should return the path where 'suppressed.csv' is saved

            In the main function, run suppression() function you implemented to suppress df.
            How many rows are unique in the suppressed data set? 
'''
# TODO: Write a function suppression() which input a DataFrame df, and a list of column names 
#       to drop these columns from df. Then save the new data set under data directory as 
#       'suppressed.csv' without a header, without index.

def suppression(df, column_names):
    """
    input: df: a DataFrame to suppress. 
            column_names: a list of strings
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
            a data set is k-anonymous for a list of quasi-identifier. 
            
            Test if the suppressed data set is k-anonymous.
'''

def is_k_anonymous(k, df, qsi):
    '''
    k: a constant to descide if a data set is k anonymous
    df: the data set of interest
    qsi: a list of quasi-identifier
    return: True if df is k-anonymous, False if not
    '''
    # TODO: inplement this function here
   


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
    Question1:  Use pandas functions to print out DataFrame information 
                and answer questions in your writeup pdf file.
                Add column names to the DataFram df. How many rows are there in df?
                Is there any personal Identifiable information in the data set?
                Can you find possible quasi-identifier among the columns? Why are they?
    '''

    # TODO:      Add column names to this data set. The list of column names are: ['age', 'workclass', 
    #           'fnlwgt', 'education', 'education_num', 'marital_status', 'moving', 'relationship', 
    #            # 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country','income']


    # TODO:      How many rows are there in this data set?



    # TODO      Write some code to explore the DataFrame.
    #           Can you find possible quasi-identifier among the columns? 
    #           Why do you think they can be quasi-identifier?




    """
    Question 2: How many rows in suppressed data set are unique?
    """
    drop_column_names = ['fnlwgt', 'education', 'relationship', 'capital_gain', 'capital_loss', 'hours_per_week']
    # TODO: Use suppression() function to suppress df data set. 
 

    """
    Question 3
    """
    ATTACK_PATH = os.path.join('data', 'link_attack.csv')
    link_attack_df = pd.read_csv(ATTACK_PATH)
    qsi = ['age','sex', 'native_country']
    # TODO: Run link_attack you implemented above in the main function with the suppressed 
    #       DataFrame, the attack DataFrame. Write code to find how many people can you re-identify? 
    #       You can use ['age',  'sex, 'native_country'] as the quasi-identifiers.


    """
    Question 4: Use is_k_anonymous(k, df, qsi)to test if the suppressed data set is 2 anonymous.
                 Use ['age', 'sex','native_country'] as quasi-identifiers.
    """
    # TODO: test if the suppressed DataFrame you generated in Question 2 is k-anonymous.
  

    '''
    Question 5: Now we wish to generalize the informations in adult.csv to make it k-anonymous while minimising
                the lose of information as little as possible. This problem turns out to be NP-Hard, which means 
                there is no polynomial algorithm to solve it. However, a compariably fast Mondrian algorithm can 
                solve it sub-optimal. The detail of this algorithm can be found at: https://github.com/qiyuangong/Mondrian
                Read through it, and run anonymizer to make this data set 2-anonymous.
    '''
    cmd = 'python3 anonymizer.py s a 2'

    # TODO: Use os module to run cmd command. 
    #       Read the anonymized data generated by this Mondrian implementation and save it in anony variable.
    #        Then add column names ['age', 'workclass',  'education_num','marital_status', 'moving', 'race', 
    #       'sex','native_country','income'] and observe this data set.
    #       Test if it is k-anonymous using the function in Question 3 and same quasi-identifier. 



    # TODO: How many rows in this data set are unique now? We can call the non-duplicate data set
    #       as unique anony.

  

    '''
    It is 2-anonymous now. And there are 0 unique rows with qsi = ['age', 'sex', 'native_country']
    '''



if __name__ == "__main__":
    main()
 