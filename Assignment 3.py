#Assignment - 3
#### DECISION TREE

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# loading data file
fullfilename = 'C:/Users/ramesh/Desktop/data2.csv'
balance_data = pd.read_csv(fullfilename,sep=',',header=0)
new_df = pd.read_csv('C:/Users/ramesh/Desktop/data2.csv', low_memory=False)

# To find the length of the database
print("Dataset Length ::", len(balance_data))

# To find the no.of lines and columns 
print("Dataset shape ::", balance_data.shape)

# Using pandas print statement to get the first 5 lines(rows) of data in the dataset
print ("Dataset:: ")
balance_data.head()

# Seperating the Target variable
X = balance_data.values[:, 1:5]
Y = balance_data.values[:, 0]

# Spliting Dataset into Test and Train
X_train, X_test, y_train, y_test = train_test_split( X,Y, test_size = 0.3, random_state = 100)

# Function to perform training with Entropy

# With maximum 4 layers, max_depth determines the no. of layers
clf_entropy_1 = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 4, min_samples_leaf = 10) 
clf_entropy_1.fit(X_train, y_train)

# With maximum 5 layers, max_depth determines the no. of layers
clf_entropy_2 = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 5, min_samples_leaf = 10)
clf_entropy_2.fit(X_train, y_train)

# Function to make Predictions

# For 4 layers
y_pred_en_1 = clf_entropy_1.predict(X_test)
print (y_pred_en_1)

# For 5 layers
y_pred_en_2 = clf_entropy_2.predict(X_test)
print (y_pred_en_2)

#Checking accuracy
print ("Accuracy is", accuracy_score(y_test,y_pred_en_1)*100)
print ("Accuracy is", accuracy_score(y_test,y_pred_en_2)*100)
