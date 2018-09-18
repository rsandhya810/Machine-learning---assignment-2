# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:15:32 2018

@author: Ramesh
"""
# Formula for calculating operating profit 
#Operating Profit = Operating Revenue - Cost of Goods Sold (COGS) - Operating Expenses - Depreciation - Amortization

import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

dataset = pd.read_csv('/Users/ramesh/Desktop/data2.csv') 

#To drop the column with more than 70% missings
df = dataset.loc[:, dataset.isnull().sum() < 0.7*dataset.shape[0]]
df


# To remove non-numerical values
reader = csv.reader(open('/Users/ramesh/Desktop/data2.csv', "rb"), delimiter=",", quotechar='"')
for line in reader:
     fields = line.split('\t')
     if fields[0].isdigit():
            df.append(fields)
            print(line)                
     
# METHOD - 1
            
train=pd.read_csv('/Users/ramesh/Desktop/data2.csv')
test=pd.read_csv('/Users/ramesh/Desktop/data2.csv')
train['Type']='Train' #Create a flag for Train and Test Data set
test['Type']='Test'
fullData = pd.concat([train,test],axis=0) #Combined both Train and Test Data set


fullData.columns # This will show all the column names
fullData.head(10) # Show first 10 records of dataframe
fullData.describe() #You can look at summary of numerical fields by using describe() function

num_cols= list(set(list(fullData.columns))
fullData.isnull().any() #Will return the feature with True or False,True means have missing value else False

num_sum = num_cols

#Impute numerical missing values with median
fullData[num_cols] = fullData[num_cols].fillna(fullData[num_cols].median(),inplace=True)

#Impute categorical missing values with -9999
fullData[cat_cols] = fullData[cat_cols].fillna(value = -9999)

#create label encoders for categorical features
for var in cat_cols:
 number = LabelEncoder()
 fullData[var] = number.fit_transform(fullData[var].astype('str'))

#Target variable is also a categorical so convert it
fullData["Account.Status"] = number.fit_transform(fullData["Account.Status"].astype('str'))

train=fullData[fullData['Type']=='Train']
test=fullData[fullData['Type']=='Test']

train['is_train'] = np.random.uniform(0, 1, len(train)) <= .05
Train, Validate = train[train['is_train']==True], train[train['is_train']==False]

features=list(set(list(fullData.columns))

x_train = Train[list(features)].values
y_train = Train["Account.Status"].values
x_validate = Validate[list(features)].values
y_validate = Validate["Account.Status"].values
x_test=test[list(features)].values

random.seed(100)
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)

status = rf.predict_proba(x_validate)
fpr, tpr, _ = roc_curve(y_validate, status[:,1])
roc_auc = auc(fpr, tpr)
print roc_auc

final_status = rf.predict_proba(x_test)
test["Account.Status"]=final_status[:,1]
test.to_csv('/Users/ramesh/Desktop/data2.csv')



#   METHOD - 2

#To fit the model

X = pd.DataFrame(df.data, columns=df.feature_names)
y = df.target


def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
  
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

result = stepwise_selection(X, y)

print('resulting features:')
print(result)











