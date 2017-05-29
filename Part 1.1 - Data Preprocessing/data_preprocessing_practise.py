# -*- coding: utf-8 -*-
"""
Created on Fri May 26 20:13:09 2017

@author: Hasib Iqbal
"""

# Importing Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

# Importing Dataset

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#Handling missing data by putting mean value

from sklearn.preprocessing import Imputer
#ctrl + i to see information
imputer = Imputer(missing_values = 'NaN',strategy="mean",axis=0)
imputer = imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:,1:3])


#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_x = LabelEncoder()

x[:,0]= labelEncoder_x.fit_transform(x[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

#Splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)