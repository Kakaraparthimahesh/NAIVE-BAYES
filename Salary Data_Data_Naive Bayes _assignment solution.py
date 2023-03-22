# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 01:01:11 2022

@author: MAHESH
"""
# DATA PROCESSING 

import pandas as pd
SalaryData_Test_data = pd.read_csv("SalaryData_Test.csv")
SalaryData_Train_data = pd.read_csv("SalaryData_Train.csv")

SalaryData_data = pd.concat([SalaryData_Test_data,SalaryData_Train_data])
SalaryData_data

SalaryData_data.shape
SalaryData_data.head()
list(SalaryData_data)
SalaryData_data.info()
SalaryData_data.isnull().sum()
SalaryData_data.describe()

# <<<<< EXPLORATION DATA ANALYSIS <<<<<

# histograme 

SalaryData_data.hist()

# box plot 

SalaryData_data.boxplot()

# columns names

SalaryData_data.columns

# LabelEncoder

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

SalaryData_data.iloc[:,5]
for eachcolumn in range(0,13):
    SalaryData_data.iloc[:,eachcolumn] = LE.fit_transform(SalaryData_data.iloc[:,eachcolumn])

SalaryData_data.head()

# split as X and Y vairables

X = SalaryData_data.iloc[:,0:12]
list(X)

Y  = SalaryData_data['Salary']

# model development

from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X,Y)

Y_pred = MNB.predict(X)

# confusion matrix and accuracy

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(Y,Y_pred)
acc = accuracy_score(Y,Y_pred).round(2)
print("naive bayes model accuracy score:" , acc)

X[0:12]
MNB.predict(X[0:12])
X.shape
X.iloc[452220,]

MNB.predict([X.iloc[45220,]])

# >>>>>>>>>>>>>>>>>>>>>>>> RESULT <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# THE ACCURACY SCORE OF THE ABOVE MODEL IS 78%

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


