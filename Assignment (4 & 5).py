# -*- coding: utf-8 -*-
"""
Created on Tue Nov 3 14:12:48 2018

@author: Ramesh
"""

############################ Assignment 4 #####################################

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

############################ Assignment 5 #####################################
import xgboost                                                                      
from sklearn.ensemble import RandomForestRegressor     
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mse as mse                 
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mse as mse                 
from sklearn import metrics

# 1) To impute missing values with median
merge_data = merge_data.fillna(merge_data.median())

# 2) To keep only numerical variables
merge_data = merge_data.select_dtypes(include = ['number'])

# 3) To remove missing variables
merge_data = merge_data.dropna(axis = 1, how = 'any')
merge_data

# 4) To divide the dataset to test and train
# Where test data = 2016, 2017, and 2018 and the rest as train

# For test data (2016-2018)
test = merge_data[(merge_data['Year'] >= 2016) & (merge_data['Year'] <= 2018)]
X_test = test.drop('spread5y', axis = 1)
Y_test = test['spread5y']

# For train data (below 2016)
train = merge_data[(merge_data['Year'] < 2016)]
X_train = train.drop('spread5y', axis = 1)
Y_train = train['spread5y']

# 5) On the train sample run a Random Forest with 50 trees
RF = RandomForestRegressor(n_estimators = 50) 
RF.fit(X_train, Y_train) 

# To predict and score the model 
Pre = RF.predict(X_test)
errors = abs(Pre - Y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2))
print('Mean Accuracy:', RF.score(X_test,Y_test))


# To calculate mean absolute percentage error
from sklearn.utils import check_arrays
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true, y_pred)
    if np.array(y_true): 
        y_true, y_pred = np.array(y_true, y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mean_absolute_percentage_error(y_true, y_pred)

# 6) Keep the first 50 features with highest feature importance and discard the rest from both test and train

imp_feature = RF.feature_importances_
imp_feature = pd.DataFrame(RF.feature_importances_, index = X_train.columns, 
                      columns = ['importance']).sort_values('importance',
                                ascending = False)

new_feature = imp_feature.iloc[:50, :]
new_feature = new_feature.index.tolist()

# To keep the first 50 important features
X_train_imp = X_train[list(new_feature)]
X_test_imp = X_test[list(new_feature)]

#########################

# 7) To run Random Forest for the 50 variables with 100, 200, 500, and 1000 trees


# For no.of trees = 100
RF_1 = RandomForestRegressor(n_estimators = 100,max_depth = 4)
RF_1.fit(X_train_imp,Y_train)
pred1 = RF_1.predict(X_test_imp)

print('Mean Accuracy_1:',  RF_1.score(X_test_imp, Y_test))
print('Mean Squared Error_1:', RF_1.score(X_test_imp, Y_test))  
print('Root Mean Squared Error_1:', np.sqrt(RF_1.score(X_test_imp, Y_test))) 

# For no.of trees = 200
RF_2 = RandomForestRegressor(n_estimators = 200,max_depth = 4)
RF_2.fit(X_train_imp,Y_train)
pred2 = RF_2.predict(X_test_imp)

print('Mean Accuracy_2:', RF_2.score(X_test_imp,Y_test))
print('Mean Squared Error_2:', RF_2.score(X_test_imp, Y_test))  
print('Root Mean Squared Error_2:', np.sqrt(RF_2.score(X_test_imp, Y_test)))

# For no.of trees = 500
RF_3 = RandomForestRegressor(n_estimators = 500,max_depth = 4)
RF_3.fit(X_train_imp,Y_train)
pred3 = RF_3.predict(X_test_imp)

print('Mean Accuracy_2:', RF_3.score(X_test_imp,Y_test))
print('Mean Squared Error_3:', RF_3.score(X_test_imp, Y_test))  
print('Root Mean Squared Error_3:', np.sqrt(RF_3.score(X_test_imp, Y_test)))

# For no.of trees = 1000
RF_4 = RandomForestRegressor(n_estimators = 500,max_depth = 4)
RF_4.fit(X_train_imp,Y_train)
pred4 = RF_4.predict(X_test_imp)

print('Mean Accuracy_2:', RF_4.score(X_test_imp,Y_test))
print('Mean Squared Error_4:', RF_4.score(X_test_imp, Y_test))  
print('Root Mean Squared Error_4:', np.sqrt(RF_4.score(X_test_imp, Y_test)))

# 7) To run Gradient Boosting & XGBoost for the 50 variables with 100, 200, 500, and 1000 trees

# For no.of trees = 100
Gradient_Boosting_1 = GradientBoostingRegressor(n_estimators = 100, max_features=None, 
                                max_depth = 4,max_leaf_nodes=None,min_samples_leaf=1,
                                min_samples_split=2, min_weight_fraction_leaf=0.0,
                                subsample=1.0, verbose=0, warm_start=False )
Gradient_Boosting_1.fit(X_train_imp, Y_train)
mse1 = mean_squared_error(Y_test, Gradient_Boosting_1.predict(X_test_imp))

XGBoost_1 = xgboost.XGBRegressor(n_estimators=100, max_features=None, max_depth = 4,
                                max_leaf_nodes=None,min_samples_leaf=1,
                                min_samples_split=2,learning_rate=0.1, min_weight_fraction_leaf=0.0,
                                subsample=1.0, verbose=0, warm_start=False)
XGBoost_1.fit(X_train_imp, Y_train)
XGB_mse_1 = mean_squared_error(Y_test, XGBoost_1.predict(X_train_imp))

# 8) To print the MSE for GBR and XGB
print("Mean Squared Error_1: %.5f" % mse1)
print("XGB_Mean Squared Error_1: %.5f" % XGB_mse_1)


# For no.of trees = 200
Gradient_Boosting_2 = GradientBoostingRegressor(n_estimators = 200, max_features=None, 
                                max_depth = 4,max_leaf_nodes=None,min_samples_leaf=1,
                                min_samples_split=2, min_weight_fraction_leaf=0.0,
                                subsample=1.0, verbose=0, warm_start=False )
Gradient_Boosting_2.fit(X_train_imp, Y_train)
mse_2 = mean_squared_error(Y_test, Gradient_Boosting_2.predict(X_test_imp))

XGBoost_2 = xgboost.XGBRegressor(n_estimators=200, max_features=None, max_depth = 4,
                                max_leaf_nodes=None,min_samples_leaf=1,
                                min_samples_split=2,learning_rate=0.1, min_weight_fraction_leaf=0.0,
                                subsample=1.0, verbose=0, warm_start=False)
XGBoost_2.fit(X_train_imp, Y_train)
XGB_mse_2 = mean_squared_error(Y_test, XGBoost_2.predict(X_train_imp))

# 8) To print the MSE for GBR and XGB
print("Mean Squared Error_2: %.5f" % mse_2)
print("XGB_Mean Squared Error_2: %.5f" % XGB_mse_2)



# For no.of trees = 500
Gradient_Boosting_3 = GradientBoostingRegressor(n_estimators = 500,  max_features=None, 
                                max_depth = 4,max_leaf_nodes=None,min_samples_leaf=1,
                                min_samples_split=2, min_weight_fraction_leaf=0.0,
                                subsample=1.0, verbose=0, warm_start=False )
Gradient_Boosting_3.fit(X_train_imp, Y_train)
mse_3 = mean_squared_error(Y_test, Gradient_Boosting_3.predict(X_test_imp))

XGBoost_3 = xgboost.XGBRegressor(n_estimators=500, max_features=None, max_depth = 4,
                                max_leaf_nodes=None,min_samples_leaf=1,
                                min_samples_split=2,learning_rate=0.1, min_weight_fraction_leaf=0.0,
                                subsample=1.0, verbose=0, warm_start=False)
XGBoost_3.fit(X_train_imp, Y_train)
XGB_mse_3 = mean_squared_error(Y_test, XGBoost_3.predict(X_train_imp))

# 8) To print the MSE for GBR and XGB
print("Mean Squared Error_3: %.5f" % mse_3)
print("XGB_Mean Squared Error_3: %.5f" % XGB_mse_3)


# For no.of trees = 1000
Gradient_Boosting_4 = GradientBoostingRegressor(n_estimators = 1000, max_features=None, 
                                max_depth = 4,max_leaf_nodes=None,min_samples_leaf=1,
                                min_samples_split=2, min_weight_fraction_leaf=0.0,
                                subsample=1.0, verbose=0, warm_start=False )
Gradient_Boosting_4.fit(X_train_imp, Y_train)
mse_4 = mean_squared_error(Y_test, Gradient_Boosting_4.predict(X_test_imp))

XGBoost_4 = xgboost.XGBRegressor(n_estimators=1000, max_features=None, max_depth = 4,
                                max_leaf_nodes=None,min_samples_leaf=1,
                                min_samples_split=2,learning_rate=0.1, min_weight_fraction_leaf=0.0,
                                subsample=1.0, verbose=0, warm_start=False)
XGBoost_4.fit(X_train_imp, Y_train)
XGB_mse_4 = mean_squared_error(Y_test, XGBoost_4.predict(X_train_imp))

# 8) To print the MSE for GBR and XGB
print("Mean Squared Error_4: %.5f" % mse_4)
print("XGB_Mean Squared Error_4: %.5f" % XGB_mse_4)




