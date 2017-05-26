# Basic-Machine-Learning-with-Python-and-R

## Part 1.2 - Data Preprocessing 

Data pre-processing is an important step in the data mining process. The phrase"garbage in, garbage out"is particularly applicable to data mining andmachine learningprojects. Data-gathering methods are often loosely controlled, resulting inout-of-rangevalues (e.g., Income: ?100), impossible data combinations (e.g., Sex: Male, Pregnant: Yes),missing values, etc. Analyzing data that has not been carefully screened for such problems can produce misleading results. Thus, the representation andquality of datais first and foremost before running an analysis. 

If there is much irrelevant and redundant information present or noisy and unreliable data, thenknowledge discoveryduring the training phase is more difficult. Data preparation and filtering steps can take considerable amount of processing time. Data pre-processing includescleaning,Instance selection,normalization,transformation,feature extractionandselection, etc. The product of data pre-processing is the finaltraining set. Kotsiantis et al. (2006) present a well-known algorithm for each step of data pre-processing. 

Data Preparation = Data Cleansing + Feature Engineering 



## Part 2.2  Regression 

Regression models (both linear and non-linear) are used for predicting a real value, like salary for example. If yourindependent variable is time, then you are forecasting future values, otherwise your model is predicting present butunknown values. Regression technique vary from Linear Regression to SVR and RandomForests Regression. 
In this part, you will understand and learn how to implement the following Machine Learning Regression models: 
1. Simple Linear Regression 
2. Multiple Linear Regression 
3. Polynomial Regression 
4. Support Vector for Regression (SVR) 
5. Decision Tree Classification 
6. Random Forest Classification 
	 
 ### 1. Simple Linear Regression 
   -------------------------------
   ![screenshot from 2017-05-27 00-07-36](https://cloud.githubusercontent.com/assets/15044221/26509615/56b55336-427b-11e7-9a25-4dfb295a7a08.png)
   ![screenshot from 2017-05-27 00-11-50](https://cloud.githubusercontent.com/assets/15044221/26509717/cc5ccae2-427b-11e7-94f6-6675965fc4c0.png)
   ![4](https://cloud.githubusercontent.com/assets/15044221/26510293/708f2266-427e-11e7-88df-9811b75ec53c.png)

```
# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
 

```

