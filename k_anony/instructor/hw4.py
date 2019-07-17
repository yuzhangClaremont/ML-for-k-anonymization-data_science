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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# def performance(y_true, y_pred, metric="f1_score") :
#     """
#     Calculates the performance metric based on the agreement between the 
#     true labels and the predicted labels.
    
#     Parameters
#     --------------------
#         y_true  -- numpy array of shape (n,), known labels
#         y_pred  -- numpy array of shape (n,), (continuous-valued) predictions
#         metric  -- string, option used to select the performance measure
#                    options: 'accuracy', 'f1_score', 'auroc', 'precision',
#                             'sensitivity', 'specificity'        
    
#     Returns
#     --------------------
#         score   -- float, performance score
#     """
#     # map continuous-valued predictions to binary labels
#     y_label = np.array(y_pred)
#     y_label[y_label==0] = 1 # map points of hyperplane to +1
    
#     # compute performance
#     if metric == "accuracy" :
#         return metrics.accuracy_score(y_true, y_label)
#     elif metric == "f1_score" :
#         return metrics.f1_score(y_true, y_label, average="weighted")
#     elif metric == "auroc" :
#         return metrics.roc_auc_score(y_true, y_pred)
#     elif metric == "precision" :
#         return metrics.precision_score(y_true, y_label)
#     elif metric == "sensitivity" :
#         #conf_mat = metrics.confusion_matrix(y_true, y_label, [1,-1])
#         #tp = conf_mat[0,0]
#         #fn = conf_mat[0,1]
#         #return np.float64(tp)/(tp+fn)
#         return metrics.recall_score(y_true, y_label)
#     elif metric == "specificity" :
#         conf_mat = metrics.confusion_matrix(y_true, y_label, [1,-1])
#         tn = conf_mat[1,1]
#         fp = conf_mat[1,0]
#         return np.float64(tn)/(tn+fp)

# def performance_CI(y_true, y_pred, metric="f1_score") :
#     """
#     Estimates the performance of the classifier using the 95% CI.
    
#     Parameters
#     --------------------
#         y            -- numpy array of shape (n,), binary labels {0,1} of test set
#         y_pred       -- numpy array of shape (n,), probability predictions for test set
#         metric       -- string, option used to select performance measure
    
#     Returns
#     --------------------
#         score        -- float, classifier performance
#         lower        -- float, lower limit of confidence interval
#         upper        -- float, upper limit of confidence interval
#     """
    

#     score = performance(y_true, y_pred, metric)
    
#     # run bootstraps
#     n = len(y_true)
#     score_list = []
#     for trial in range(1000) :
#         sample = np.random.randint(n, size=n)
#         score_b = performance(y_true[sample], y_pred[sample], metric)
#         if not np.isnan(score_b) :
#             score_list.append(score_b)
    
#     # compute CI
#     score_list.sort()
#     l = len(score_list)
#     lower = score_list[int(0.025*l)]
#     upper = score_list[int(0.975*l)]
    
#     return score, lower, upper

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


def main():

    PATH = os.path.join('data', 'adult.data')
    # PATH = os.path.join('data', 'monAdult.csv'
    df = pd.read_csv(PATH)
    

    '''
    Question1:  In the main function of hw4.py, use pandas functions to print out DataFrame information 
                and answer questions in your writeup pdf file.

                Is there any personal Identifiable information in the data set?
                Is there any quasi-identifier information according to Sweeney's defination?
                Can you find possible quasi-identifier among the columns? Why are they?

                Write some codes after TODO. Then write your response in a seperate file and save it as hw4.pdf.
    '''
 
    # TODO:     A. How many rows are there in this data set?
    print('there are', df.shape[0], 'rows in df~~~')
    '''
    there are 32560 rows in the data set  
    '''

    # TODO      C. Write some code to explore the DataFrame.
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
    Question 2 Data preprocess and Feature Engineering

    """
    missing_values = ["n/a", "na", "--", "unknown", " ?","", ' ']
    # TODO: A. Data clean
    df = pd.read_csv(PATH, na_values = missing_values)
    df.dropna(axis = 0, how = 'any', inplace = True)
    print('there are', df.shape[0], 'rows in cleaned df~~~')
    # -> 30161 after drop na

    # TODO:     B. Add column names to this data set. The list of column names are: ['age', 'workclass', 
    #           'fnlwgt', 'education', 'education_num', 'marital_status', 'moving', 'relationship', 
    #            # 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country','income']
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'moving', 'relationship', 'race', 'sex',
    'capital_gain','capital_loss', 'hours_per_week', 'native_country','income']
    
    # TODO:C run suppression() function you implemented with the inputs as the DataFrame df from question 1,
    # and the column names mentioned above. 
    drop_column_names = ['fnlwgt', 'education', 'relationship', 'capital_gain', 'capital_loss', 'hours_per_week']
    suppressed_path = suppression(df, drop_column_names)
    suppressed_df = pd.read_csv(suppressed_path)
    suppressed_df.columns = ['age', 'workclass',  'education_num',
       'marital_status', 'moving', 'race', 'sex',
        'native_country','income']
    # print(df.shape[0], suppressed_df.shape[0]) # 30161 30160
    # TODO: D How many rows are unique in the suppressed data set? 
    unique = suppressed_df.drop_duplicates(keep = False)
    # print(unique.head()) 
    print(unique.describe()) 
    '''
    15512 unique
    '''

    """
    Question 3 b
    """
    ATTACK_PATH = os.path.join('data', 'link_attack.csv')

    link_attack_df = pd.read_csv(ATTACK_PATH)
    qsi = ['age','sex', 'native_country']
    # TODO: B Run link_attack you implemented above in the main function with the suppressed 
    #       DataFrame, the attack DataFrame. Write code to find how many people can you re-identify? 
    #       You can use ['age',  'sex, 'native_country'] as the quasi-identifiers.
    merged_df = link_attack(suppressed_df, link_attack_df,  qsi )
    print()

    merged_unique = merged_df.drop_duplicates(['age','sex', 'native_country'], keep = False)
    # print(merged_unique.head())
    print(merged_unique.describe(), 'can be identified by link attack') 
    '''
    280 people can be identified by link attack
    '''

    """
    Question 4: Use is_k_anonymous(k, df, qsi)to test if the suppressed data set is 2 anonymous.
                 Use ['age', 'sex','native_country'] as quasi-identifiers.
    """
    # TODO: B your code here to test if suppressed data set is k anonymous
    print('is suppressed file k anonymous? ',is_k_anonymous(2,suppressed_df, qsi))

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

    os.system(cmd) # NCP = 4.97%
    anony = pd.read_csv('data/anonymized.csv')
    anony.columns = ['age', 'workclass',  'education_num',
        'marital_status', 'moving', 'race', 'sex',
            'native_country','income']
    # print(anony.head())
    # print(anony.describe())
    print('is anony file k anonymous? ',is_k_anonymous(2,anony, qsi))


    # TODO: How many rows in this data set are unique now? We can call the non-duplicate data set
    #       as unique anony.

    unique_anony = anony.drop_duplicates(['age', 'sex', 'native_country'], keep = False)
    # print(unique_anony.head()) #
    print(unique_anony.describe()) # 0 unique

    '''
    It is 2-anonymous now. And there are 0 unique rows with qsi = ['age', 'sex', 'native_country']
    '''


    '''
    Question 6: Analize df and the k-anonymous data sets in the main function. What percentage of 
                male and female has income less than 50k? Does the process of k-anonymous make the 
                conclusion different? What are the advantages and disadvantages of k-anonymization?
    '''
    # TODO: Your analysis code here
    # sex_group = anony.groupby(['sex'])
    # for sex, sex_df in sex_group:
    #     print(sex, sex_df.shape[0])
    #     print(sex_df.income.value_counts())

        # print(sex_df.income.value_counts(normalize = True))

    # male 13664/19995 0.6834, female 7872/8923 0.8822, male~female 1117/1243 , sum = 30161


    # print('original data!!!!!!!')
    # sex_group2 = suppressed_df.groupby(['sex'])
    # for sex, sex_df in sex_group2:
    #     print(sex, sex_df.shape[0])
    #     print(sex_df.income.value_counts( ))
    # print("total data")
    # print(anony.income.value_counts())
    # print(suppressed_df.income.value_counts())
    # print(df.income.value_counts())

        # print(sex_df.income.value_counts( normalize= True))

     # male 13982/20378 0.6861, female 8670/9782 0.8863 sum = 30160

    # print(df.isnull().sum() )
    # print(df.describe() )
    # print(anony.describe())

    # ML Pipeline

    # train, test = train_test_split(suppressed_df, test_size = 0.15)
    # suppressed data set numeric encoding

    # Create a label encoder object
    le = LabelEncoder()



    # Iterate through the columns

    # for col in train:
    #     if train[col].dtype == 'object':
    #         le.fit(suppressed_df[col])
    #         # Transform both training and testing data
    #         train[col] = le.transform(train[col])
    #         test[col] = le.transform(test[col])



    # # one-hot encoding of categorical variables
    # train = pd.get_dummies(train)
    # test = pd.get_dummies(test)
    suppressed_df = pd.get_dummies(suppressed_df, columns = ['workclass',  
        'marital_status', 'moving', 'race', 'sex', 'native_country'])
    train, test = train_test_split(suppressed_df, test_size = 0.15)
    print('Training Features shape: ', train.shape)
    print('Testing Features shape: ', test.shape)

    #  train model

    clf = DecisionTreeClassifier(min_samples_split = 100)
    # features = ['age',  'education_num','race', 'sex',]
    features = suppressed_df.columns.tolist()
    print(features)
    features.remove('income')


    x_train = train[features] 
    y_train = train['income']
    x_test = test[features]
    y_test = test['income']
 
    dt = clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)

    score = metrics.accuracy_score(y_test, y_pred)
    print('suppressed accuracy') 
    print(score)
                                                    
    p, r, f1, s = metrics.precision_recall_fscore_support(y_test, y_pred,average="weighted")

    # print("PERFORMANCE FUNC", performance(np.array(y_test), np.array(y_pred), metric = "f1_score"))
    print(metrics.confusion_matrix(y_test, y_pred))
    print('precision')                                                
    print(p)
    print('recall') 
    print(r)
    print('f1') 
    print(f1)
    score, lower, upper = performance_CI(np.array(y_test), np.array(y_pred) )
    print('confidence interval')
    print(score, lower, upper)

    #   k-anonymized data

    train, test = train_test_split(anony, test_size = 0.15)

    # anony data numerical encoding

    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in train:
        if train[col].dtype == 'object':
            # Train on the training data
            le.fit(anony[col])

            # Transform both training and testing data
            train.loc[:,col] = le.transform(train[col])
            test.loc[:,col] = le.transform(test[col])
            


    # # one-hot encoding of categorical variables
    # train = pd.get_dummies(train)
    # test = pd.get_dummies(test)

    # anony model fitting

    clf = DecisionTreeClassifier(min_samples_split = 100)
    # features = ['age',  'education_num','race', 'sex',]
    features = ['age', 'workclass',  'education_num',
       'marital_status', 'moving', 'race', 'sex',
        'native_country']

    x_train = train[features] 
    y_train = train['income']
    x_test = test[features]
    y_test = test['income']

    
    dt = clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)

    score = accuracy_score(y_test, y_pred)
    print('anony accuracy') 
    print(score)
                                                 
    p, r, f1, s = metrics.precision_recall_fscore_support(y_test, y_pred,
                                                          average="weighted")

    print(metrics.confusion_matrix(y_test, y_pred))
    # print("PERFORMANCE FUNC", performance(y_test, y_pred, metric = "f1_score"))
    print('precision')                                                
    print(p)
    print('recall') 
    print(r)
    print('f1') 
    print(f1)
    # score, lower, upper = performance_CI(np.array(y_test), np.array(y_pred) )
    print('confidence interval')
    print(score, lower, upper)


    # male 13685/20017, female 8002/9062
    # print('race anony data!!!!!!!')
    # race_group = anony.groupby(['race'])
    # for race, race_df in race_group:
    #     print(race)
    #     print(race_df.income.value_counts())

    # # male 13685/20017, female 8002/9062

    # print('race original data!!!!!!!')
    # race_group2 = df.groupby(['race'])
    # for race, race_df in race_group:
    #     print(race)
    #     print(race_df.describe())

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



