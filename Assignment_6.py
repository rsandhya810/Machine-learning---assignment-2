# -*- coding: utf-8 -*-
"""
Created on Mon Dec 2 23:12:08 2018

@author: Ramesh
"""
import pandas as pd
import numpy as np

#Import cds data file
data1 = pd.io.stata.read_stata('/Users/ramesh/Desktop/cds_spread5y_2001_2016.dta')

# convert cds data to csv file
data = data1.to_csv('/Users/ramesh/Desktop/cds_spread5y_2001_2016.csv')

# Import crsp data file 
data2 = pd.read_csv("/Users/ramesh/Desktop/CRSP.csv")

# Match the column names in CDS data

df1 = data2.columns = data2.columns.str.replace('GVKEY', 'gvkey')
df1

df2 = data2.columns = data2.columns.str.replace('datadate', 'mdate')
df2

df3 = data2
df3
######################## PART 1 - CDS file

# Merge the files with gvkey
# This step is done to just to lookup the merged data file on only gvkey

data1['gvkey'] = data1['gvkey'].astype(float)
merged = data1.merge(df3, on='gvkey')
merged.to_csv("merged.csv", index=False)
data3 = pd.read_csv("/Users/ramesh/Desktop/merged_data.csv")

#Splitting the year, month, day from the date column from CDS file

data1['Date'] = pd.to_datetime(data1['mdate'])
data1['Year'],data1['Month'],data1['Day'] = data1.Date.dt.year, data1.Date.dt.month, data1.Date.dt.day
data1

#Assign quarters based on no.of months
data1['Q'] = "4"

data1['Q'] [data1['month'] > 9] = "4"
data1['Q'][(data1['month']  > 6) & (data1['month'] < 9)] = "3"
data1['Q'][(data1['month']  > 3) & (data1['month'] < 6)] = "2"
data1['Q'][data1['month']  < 3] = "1"

# To convert gvkey, Quarter and Year column to float
data1['gvkey'] = data1['gvkey'].astype(float)
data1['Q'] = data1['Q'].astype(float)
data1['Year'] = data1['Year'].astype(float)
# This step is done to enable merging with the CRSP file 

###################### PART 2 - CRSP file

#Splitting the year, month, day from the date column from CRSP file

data2['Date'] = pd.to_datetime(data1['mdate'])
data2['Year'],data2['Month'],data2['Day'] = data2.Date.dt.year, data2.Date.dt.month, data2.Date.dt.day
data2

#Assign quarters based on no.of months
data2['Q'] = "4"

data2['Q'] [data2['Month'] > 9] = "4"
data2['Q'][(data2['Month']  > 6) & (data2['Month'] < 9)] = "3"
data2['Q'][(data2['Month']  > 3) & (data2['Month'] < 6)] = "2"
data2['Q'][data2['Month']  < 3] = "1"

# To convert gvkey, Quarter and Year column to float
data2['gvkey'] = data2['gvkey'].astype(float)
data2['Q'] = data2['Q'].astype(float)
data2['Year'] = data2['Year'].astype(float)
# This step is done to enable merging with the CRSP file 

# To merge the columns selected from both the data files (CDS & CRSP)
merge_data = pd.merge(data1, data2, on = ['gvkey', 'Q', 'Year'])
merge_data
merge_data.to_csv("merged_output.csv", index=False)
###############################################################################

#####ASSIGNMENT - 6
import xgboost                                                                      
from sklearn.ensemble import RandomForestRegressor     
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mse as mse                 
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mse as mse                 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import numpy

# To impute missing values with median
merge_data = merge_data.fillna(merge_data.median())

# To keep only numerical variables
merge_data = merge_data.select_dtypes(include = ['number'])

# To remove missing variables
merge_data = merge_data.dropna(axis = 1, how = 'any')
merge_data

# 1. Split the sample to test and train (80% train)

X= merge_data.drop('spread5y', axis=1)
Y= merge_data ['spread5y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train

# 2. Reduce the number of variables to something about 40. Choose any reduction technique that you would like.

# Removing features from train and test dataset

X_test = X_test.drop('Month_x', axis = 1)
X_test = X_test.drop('Month_y', axis = 1)
X_test = X_test.drop('Quarter', axis = 1)
X_test = X_test.drop('Year',    axis = 1)
X_test = X_test.drop('gvkey',   axis = 1)

X_train= X_train.drop('Month_x', axis = 1)
X_train= X_train.drop('Month_y', axis = 1)
X_train= X_train.drop('Quarter', axis = 1)
X_train= X_train.drop('Year',    axis = 1)
X_train= X_train.drop('gvkey',   axis = 1)



# 3. Standardize the variables.

scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy = True, with_mean = True, with_std = True)

X_train_1 = X_train
X_test_1 = X_test

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 4.	Run a Neural Network with Grid search. In your grid search try 3 different values for 3 parameters of your network (your choice).

# First Model

RF = RandomForestRegressor(n_estimators=10) 

# Model is fit to training and scored to testindg data
RF.fit(X_train, Y_train) 
RF.score(X_test, Y_test)
R_F = RF.predict(X_test)

# To select top 40 feature importance
FI = RF.feature_importances_

FI = pd.DataFrame(RF.feature_importances_, index = X_train.columns, 
                      columns = ['importance']).sort_values('importance',
                                ascending = False)
FI_40 = FI.iloc[:40, :]
FI_40 = FI_40.index.tolist()

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_squared_error(Y_test,R_F)
mean_absolute_percentage_error(Y_test, R_F)

Filter_X_train=X_train_1[FI_40]
Filter_X_test=X_test_1[FI_40]


# Fit to the training data and filter
scaler.fit(Filter_X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
Filtered_X_train = scaler.transform(Filter_X_train)
Filtered_X_test = scaler.transform(Filter_X_test)

numpy.random.seed(7)

# To create MLP model

model1 = Sequential()
model1.add(Dense(32, input_dim=50, activation='relu'))
model1.add(Dense(8, activation='relu'))
model1.add(Dense(1, activation='linear'))

# To compile the model
model1.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

# To fit the model
model1.fit(Filter_X_train, Y_train, epochs = 100, batch_size = 10)


# Neural Network for Model 1
NN_1 = model1.predict(Filter_X_test)
mape_NN1 = mape(Y_test, NN_1)
mape_NN1


# Neural Network for Model 2
model2 = Sequential()
model2.add(Dense(32, input_dim = 50, activation = 'sigmoid'))
model2.add(Dense(8, activation = 'sigmoid'))
model2.add(Dense(1, activation = 'relu'))
model2.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

model2.fit(Filter_X_train, Y_train, epochs = 10, batch_size = 10)

# To predict the model
NN_2 = model2.predict(Filter_X_test)
mape_NN2 = mape(Y_test, NN_2)
mape_NN2

# Neural Network For Model 3
model3 = Sequential()
model3.add(Dense(32, input_dim = 50, activation = 'sigmoid'))
model3.add(Dense(8, activation = 'sigmoid'))
model3.add(Dense(1, activation = 'relu'))
model3.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

model3.fit(Filter_X_train, Y_train, epochs = 50, batch_size = 10)

NN_3 = model3.predict(Filter_X_test)
mape_NN3 = mape(Y_test, NN_3)
mape_NN3


# To print the best MAPE

print("The best MAPE as model no.3 ")