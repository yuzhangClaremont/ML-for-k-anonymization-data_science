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
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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
    input: df: a DataFrame under attack. attack_df: a DataFrame used to attack. qsi: quasi-identifiers
    output: a link-attack DataFrame that can reveal information if individual can be identified 
            by quasi-identifiers.
    """

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
    qsi: a list of quasi-identifiers
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
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain','capital_loss', 'hours_per_week', 'native_country','income']
    
    # TODO:C run suppression() function you implemented with the inputs as the DataFrame df from question 1,
    # and the column names mentioned above. 
    drop_column_names = ['fnlwgt', 'education', 'relationship', 'capital_gain', 'capital_loss', 'hours_per_week']
    suppressed_path = suppression(df, drop_column_names)
    suppressed_df = pd.read_csv(suppressed_path)
    suppressed_df.columns = ['age', 'workclass',  'education_num',
       'marital_status', 'occupation', 'race', 'sex',
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

    # TODO: A Use os module to run cmd command. 
    #       Read the anonymized data generated by this Mondrian implementation and save it in anony variable.
    #        Then add column names ['age', 'workclass',  'education_num','marital_status', 'moving', 'race', 
    #       'sex','native_country','income'] and observe this data set.
    #       Test if it is k-anonymous using the function in Question 3 and same quasi-identifier.

    os.system(cmd) # NCP = 4.97%
    anony = pd.read_csv('data/anonymized.csv')
    anony.columns = ['age', 'workclass',  'education_num',
        'marital_status', 'occupation', 'race', 'sex',
            'native_country','income']
    # print(anony.head())
    # print(anony.describe())
    print('is anony file k anonymous? ',is_k_anonymous(2,anony, qsi))


    # TODO: B How many rows in this data set are unique now? We can call the non-duplicate data set
    #       as unique anony.

    unique_anony = anony.drop_duplicates(['age', 'sex', 'native_country'], keep = False)
    # print(unique_anony.head()) #
    print(unique_anony.describe()) # 0 unique

    '''
    It is 2-anonymous now. And there are 0 unique rows with qsi = ['age', 'sex', 'native_country']
    '''


    '''
    Question 6: Analize df and the k-anonymous data sets in the main function. What percentage of 
                male and female has income less than 50k? Build a decision tree model for both data 
                sets and evaluate their performance. Use ['age', 'workclass',  'education_num',
                'marital_status', 'occupation', 'race', 'sex','native_country']
                features to build decision tree models to predict if individual's income is less than
                50k or more than 50k. Use both the suppressed dataset and k-anonymous 
                dataset.
    '''
    # TODO A: Your analysis code here
    print('Question 6A original data')
    sex_group2 = suppressed_df.groupby(['sex'])
    for sex, sex_df in sex_group2:
        print(sex, sex_df.shape[0])
        print(sex_df.income.value_counts( ))
    # male 13982/20378 0.6861, female 8670/9782 0.8863 sum = 30160
    print('Question 6A k-anonymous data')
    sex_group = anony.groupby(['sex'])
    for sex, sex_df in sex_group:
        print(sex, sex_df.shape[0])
        print(sex_df.income.value_counts())
        print(sex_df.income.value_counts(normalize = True))
    # male 13664/19995 0.6834, female 7872/8923 0.8822, male~female 1117/1243 , sum = 30161

    # Part B: Use ['age', 'workclass',  'education_num','marital_status', 'occupation', 'race',
    # 'sex','native_country'] features to build decision tree models to predict if individual's 
    # income is less than 50k or more than 50k. Use both the suppressed dataset and k-anonymous 
    #  dataset.

    # ML Pipeline
    # split suppressed dataset to train and test parts
    train, test = train_test_split(suppressed_df, test_size = 0.2)

    # Create a label encoder object for numeric encoding
    le = LabelEncoder()
    # Iterate through the columns
    for col in train:
        if train[col].dtype == 'object':
            le.fit(suppressed_df[col])
            # Transform both training and testing data
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])

    print(train.income.value_counts())

    #  train model
    clf = DecisionTreeClassifier( max_depth = 7,min_samples_split = 0.01) #f1 = 0.817
    # clf = DecisionTreeClassifier() # 0.781
    features = ['age', 'workclass',  'education_num',
       'marital_status', 'occupation', 'race', 'sex',
        'native_country']

    x_train = train[features] 
    y_train = train['income']
    x_test = test[features]
    y_test = test['income']
 
    # TODO: fit the decision tree model, then report f1 score for the original dataset
    dt = clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)                                               
    p, r, f1, s = metrics.precision_recall_fscore_support(y_test, y_pred,
                                                          average="weighted")
    print('f1 score:') 
    print(f1)


    # max_features = list(range(1,train.shape[1]))
    # train_results = []
    # test_results = []
    # for max_feature in max_features:
        
    #     dt = DecisionTreeClassifier(max_features=max_feature)
    #     dt.fit(x_train, y_train)
    #     train_pred = dt.predict(x_train)
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     train_results.append(roc_auc)
    #     y_pred = dt.predict(x_test)
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     test_results.append(roc_auc)

    # from matplotlib.legend_handler import HandlerLine2D
    # line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')
    # line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    # plt.ylabel('AUC score')
    # plt.xlabel('max features')
    # plt.show()

    # min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
    # train_results = []
    # test_results = []
    # for min_samples_split in min_samples_splits:

    #     dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
    #     dt.fit(x_train, y_train)
    #     train_pred = dt.predict(x_train)
    #     false_positive_rate, true_positive_rate, thresholds =    roc_curve(y_train, train_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     train_results.append(roc_auc)
    #     y_pred = dt.predict(x_test)
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     test_results.append(roc_auc)

    # from matplotlib.legend_handler import HandlerLine2D
    # line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
    # line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    # plt.ylabel('AUC score')
    # plt.xlabel('min samples split')
    # plt.show()

    # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    # roc_auc = auc(false_positive_rate, true_positive_rate)


    # max_depths = np.linspace(1, 32, 32, endpoint=True)
    # train_results = []
    # train_f1 = []
    # test_results = []
    # test_f1 = []
    # for max_depth in max_depths:
    #     dt = DecisionTreeClassifier(max_depth=max_depth)
    #     dt.fit(x_train, y_train)
    #     train_pred = dt.predict(x_train)
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     train_results.append(roc_auc)
        
    #     p, r, f1, s = metrics.precision_recall_fscore_support(y_train, train_pred,
    #                                                       average="weighted")
    #     train_f1.append(f1)
    #     # Add auc score to previous train results
        
    #     y_pred = dt.predict(x_test)
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)

    #     p, r, f1, s = metrics.precision_recall_fscore_support(y_test, y_pred,
    #                                                       average="weighted")
    #     # Add auc score to previous test results
    #     test_results.append(roc_auc)
    #     test_f1.append(f1)
    #     # print(test_f1)
    #     # print(train_f1)
    # from matplotlib.legend_handler import HandlerLine2D
    # line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
    # line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    # plt.ylabel('AUC score')
    # plt.xlabel('Tree depth')
    # plt.show() # 8 max_depth = 8
    # line3, = plt.plot(max_depths, train_f1, 'b', label="Train f1")
    # line4, = plt.plot(max_depths, test_f1, 'r', label="Test f1")
    # plt.legend(handler_map={line3: HandlerLine2D(numpoints=2)})
    # plt.ylabel('f1score')
    # plt.xlabel('Tree depth')
    # plt.show() # max depth = 10

    # print(score, lower, upper)

    # k-anonymized data set modeling
    # split data set
    train, test = train_test_split(anony, test_size = 0.2)

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

    clf = DecisionTreeClassifier( max_depth = 7, min_samples_split = 0.01) # f1 = 0.799
    features = ['age', 'workclass',  'education_num',
       'marital_status', 'occupation', 'race', 'sex',
        'native_country']

    x_train = train[features] 
    y_train = train['income']
    x_test = test[features]
    y_test = test['income']

    # TODO: fit the decision tree model, then report f1 score
    dt = clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)                                            
    p, r, f1, s = metrics.precision_recall_fscore_support(y_test, y_pred,
                                                          average="weighted")

    print(metrics.confusion_matrix(y_test, y_pred))

    print('k-anonymous data set decision tree f1') 
    print(f1)


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





