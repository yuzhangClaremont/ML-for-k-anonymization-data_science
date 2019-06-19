# HW4: Data Privacy and Anonymity

Readlist:
How unique are you? And explanation.
K-anonymity and Sweeney paper
Link attack (slide 8) and solution to it.
Personal identifiable information (PII)

Background:

Differential privacy is a constraint on the algorithms used to publish aggregate information about a statistical database which limits the privacy impact on individuals whose information is in the database. For example, differentially private algorithms are used by some government agencies to publish demographic information or other statistical aggregates while ensuring confidentiality of survey responses, and by companies to collect information about user behavior while controlling what is visible even to internal analysts. Roughly, an algorithm is differentially private if an observer seeing its output cannot tell if a particular individual's information was used in the computation.
One way to ensure differential privacy is to k-anonymize the data set. A release of data is said to have the k-anonymity property if the information for each person contained in the release cannot be distinguished from at least k - 1 individuals whose information also appear in the release. K-anonymization is a technique to generalize, modify, or distort quasi-identifier values so that no individual is uniquely identifiable from a group of k.

	This assignment uses a data set from the real world to show even a combination of very common quasi-identifiers can be used to re-identify individuals. Additionally, we can use link attack to retrieve sensitive information of re-identified individuals. Finally, we can perform k-anonymization to de-identify the data set without much influence of analysis utility.

Goas:
	To understand differential privacy and k-anonymity.
	To practice pandas analysis functions.

Data set:
The data set is from 1994 Census database 
retrieved from:https://archive.ics.uci.edu/ml/datasets/adult.

For this assignment, you need to use Python pandas functions to analyze the “adult.data” data set, which is in the “data” directory. You can print the analysis out, and then write response questions.

Question 1: Data observation 
Use pandas read_csv function to read adult.data. Add column names to this data set. The list of column names are: ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'moving', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country','income']

How many rows are there in this data set?
Is there any personal Identifiable information in the data set?
Is there any quasi-identifier information according to Sweeney's definition?
Can you find possible quasi-identifier among the columns? Why are they?

Useful pandas data observation functions: 
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shape.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html


Question 2: Suppression de-identification
Your supervisor believes the information in ['fnlwgt', 'education', 'relationship', 'capital_gain', 'capital_loss', 'hours_per_week'] are not necessary for your analysis.
 Use suppression technique to de-idenitify this data set. Then save the new data set under data directory as 'suppressed.csv' without a header.
How many rows are unique in the suppressed data set?

Save to csv:
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
To drop data: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
To drop (or keep) duplicated data: http://pandas.pydata.org/pandas-docs/version/0.17/generated/pandas.DataFrame.drop_duplicates.html

Question 3: Link attack
Suppose now you have a sensitive dataframe of 10,000 rows named as "link_attack.csv", which contains personal identifiable information. Take a look at the head of this attack data,
perform a link attack and to re-identify people's name, ssn, race, and salary. You can use
['age',  'sex, 'native_country'] as quasi-identifier. How many people can you identify?
Hint: You can merge two tables together to perform link attack


Question 4: 
Consider the definition of k-anonymity, write a function to decide if a data set is k-anonymous for a list of quasi-identifier. Then test if this data set adult.csv is 2 anonymous with ['age', 'sex','native_country'] being quasi-identifiers.

Helpful functions: 
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.equals.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html

Question 5: 
Now we wish to generalize the information in adult.csv to make it k-anonymous while minimising the lose of information as little as possible. This problem turns out to be NP-Hard, which means there is no polynomial algorithm to solve it. However, a compariably fast algorithm can solve it sub-optimal. The details of this algorithm can be found at: https://github.com/qiyuangong/Mondrian