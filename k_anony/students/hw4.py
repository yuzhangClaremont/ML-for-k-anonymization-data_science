import pandas as pd
import numpy as np
import os
# import utils.tools as tools 


'''
Question1:  Use pandas to read data adult.data, a data set from 1994 Census database (retrieved
            from: https://archive.ics.uci.edu/ml/datasets/adult).
            Add column names to this data set. The list of column names are: ['age', 'workclass', 
            'fnlwgt', 'education', 'education_num', 'marital_status', 'moving', 'relationship', 
            'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country','income']
            How many rows are there in this data set? 
            Is there any personal Identifiable information in the data set?
            Is there any quasi-identifier information according to Sweeney's defination?
            Can you find possible quasi-identifier among the columns? Why are they?
'''

# TODO: Read "data\adult.data" and add columns to the data set. How many rows are there in this data set? 




# TODO: Is there any personal Identifiable information in the data set?
#       Is there any quasi-identifier information according to Sweeney's defination?
#       Can you find possible quasi-identifier among the columns? 
#       Why do you think they can be quasi-identifier?



'''
Question 2: Your supervisor believes the information in ['fnlwgt', 'education', 'relationship', 
            'capital_gain', 'capital_loss', 'hours_per_week'] are not necessary for your analysis.
            Use suppression technique to de-idenitify this data set. How many rows are unique in 
            the suppressed data set? 
            Hint: Suppression technique and other de-identify techniques can be found in the reading 
            list slides https://www2.cs.duke.edu/courses/fall12/compsci590.3/slides/lec3.pdf
'''
# TODO: Use suppression technique to de-idenitify this data set. 
#       Then save the new data set under data directory as 'suppressed.csv' without a header.
#       You can set header to False and index to False to save the data set without header and index


# TODO: How many rows in suppressed data set are unique?



'''
Question 3: Suppose now you have a sensitive data set of 10,000 rows named as "link_attack.csv",
            which contains personal identifiable information. Take a look at the head of this attack data,
            perform a link attack and to re-identify people's name, ssn, race, and income. 
            Hint: You can use ['age',  'sex, 'native_country'] as quasi-identifier. 
            How many people can you identify?
 
'''
# TODO: Perform link attack and identify individuals. How many of them can you idenity?


'''
Question 4: Consider the definition of k-anonymity, write a function to decide if 
            a data set is k-anonymous for a list of quasi-identifier. Then test if 
            this data set adult.csv is 2 anonymous with ['age', 'sex','native_country']
            being quasi-identifiers
'''
qsi = ['age', 'sex', 'native_country']

def is_k_anonymous(k, df, qsi):
    '''
    k: a constant to descide if a data set is k anonymous
    df: the data set of interest
    qsi: a list of quasi-identifier
    return: True if df is k-anonymous, False if not
    '''
    # TODO: Your code here



'''
Question 5: Now we wish to generalize the informations in adult.csv to make it k-anonymous while minimising
            the lose of information as little as possible. This problem turns out to be NP-Hard, which means 
            there is no polynomial algorithm to solve it. However, a compariably fast algorithm can solve it
            sub-optimal. 
            This algorithm is algready implemented for you. 
            The detail of this algorithm can be found at: https://github.com/qiyuangong/Mondrian
            Read through it, and use this package to make this data set 2-anonymous.
'''

# TODO: Use os module to run command 'python3 anonymizer.py s a 2'. Read the head generated data '
#       data/anonymized.csv' then add column names ['age', 'workclass',  'education_num','marital_status',
#       'moving', 'race', 'sex','native_country','income']
#       back to the anonymized data set. Observe the data by printing out the head and describe funtion.



# TODO: Test if it is k-anonymous using the function in Question 3 and same quasi-identifier. 
#       How many rows in this data set are unique now?



'''
Question 6: Use the anonymized data set, analyze what percentage of male and female has income 
            less than 50k? Compare the result with the original datase 'adult.data'.
'''




