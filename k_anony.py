###
# author: Yun Zhang
# based course material: https://github.com/jnear/cs295-data-privacy/blob/master/homework/Homework%201.ipynb
###
import pandas as pd
import numpy as np

adult_data = pd.read_csv("adult_with_pii.csv")

# Remove Personal Identification Info[PII]
adult_anon = adult_data.drop(columns=['Name', 'SSN'])
# print(adult_anon.describe()) # 32561 people

# ## PII only
# pii = adult_data[['Name', 'DOB', 'SSN', 'Zip']]
# pii_flawed = pii.drop(range(100))

# ## Generate 10000 Personal Identifiable data
# pii10000 = pii[142:10142]

# # print(pii.describe())
# # print(pii10000.head())
# pii10000.to_csv(r'attack.csv',index=False)

# read attack data
attack_data = pd.read_csv("attack.csv")
print(attack_data.describe())

# Some birthdates occur more than once
# print( attack_data['DOB'].value_counts().head(n=20) )

# Some zip codes occur more than once
print( attack_data['Zip'].value_counts()[ attack_data['Zip'].value_counts()== 1].size) # 9069 can be identified

print( attack_data['DOB'].value_counts()[ attack_data['DOB'].value_counts()== 1].size) # 6431 can be identified

print( attack_data['DOB'].value_counts()[ attack_data['DOB'].value_counts()== 1].data )# 6431 can be identified

# print( pii.head().sort_values(ascending = False, by = 'Zip') )


# Question 1
# Using the dataframes pii and adult_anon, perform a linking attack to recover the names of as many samples in adult_anon as possible.
# How many names are you able to recover?
# dob = pii['DOB'].to_numpy()
# zipcode = pii['Zip'].to_numpy()
# attack = pii.copy()
# attack['DOB'] = dob
# attack['Zip'] = zipcode
# id_dob = attack['DOB'].value_counts().sort_values(ascending = True)
# id_dob = id_dob[id_dob == 1] 
# # print(id_dob) # 7845 identified by DOB

# id_zip = attack['Zip'].value_counts()
# id_zip = id_zip[id_zip == 1] 
# print(id_zip) # 23513 idenfied by zip

# In this cell, write code to perform the linking attack
# for i, row in enumerate(adult_anon.head().iterrows()):
#     print(i, row['DOB'])

# if adult_anon['DOB'] == pii['DOB'] and adult_anon['Zip'] == pii['Zip']:
#     print(pii['Name'])