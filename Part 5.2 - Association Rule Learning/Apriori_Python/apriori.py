#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:21:40 2017

@author: rezwan
"""
# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the data set
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
    
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
myResults = [list(x) for x in results]
myRes = []
for j in range(0,153):
    myRes.append([list(x) for x in myResults[j][2]])
   
n = 5
    
lists = [np.array([x == 1 for x in range(5) ]) for _ in range(n)]




