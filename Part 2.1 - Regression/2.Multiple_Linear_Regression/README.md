## Methods of Building Models
  1. All-in
  2. Backword Elimination
  3. Forward Selection
  4. Bidirectional Elimination
  5. Score Comparision
#### Note : No. 2,3,4 is combinely considered as Stepwise Regression

# Multiple Linear Regression With Python

## Importing the libraries
  `import numpy as np`
  
  `import matplotlib.pyplot as plt`
  
  `import pandas as pd`

## Importing the dataset
  `dataset = pd.read_csv('50_Startups.csv')`
  
  `X = dataset.iloc[:, :-1].values`
  
  `y = dataset.iloc[:, 4].values`

## Encoding categorical data
  // Whenever a dataset consists of categorical data , we need to encode that data for process . But in that case, we have to always concern about dummy variable trap.

  `from sklearn.preprocessing import LabelEncoder, OneHotEncoder`
  
  `labelencoder = LabelEncoder()`
  
  `X[:, 3] = labelencoder.fit_transform(X[:, 3])`
  
  `onehotencoder = OneHotEncoder(categorical_features = [3])`
  
  `X = onehotencoder.fit_transform(X).toarray()`

## Avoiding the Dummy Variable Trap
// Though We don't have to code for avoiding dummy variable trap as we are using library but here is a sample code to avoid dummy variable trap by eliminating one encoded column

  `X = X[:, 1:]`

## Splitting the dataset into the Training set and Test set
 ` from sklearn.cross_validation import train_test_split`
 
 ` X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)`

## Feature Scaling
  `"""from sklearn.preprocessing import StandardScaler`
  
  `sc_X = StandardScaler()`
  
  `X_train = sc_X.fit_transform(X_train)`
  
  `X_test = sc_X.transform(X_test)`
  
  `sc_y = StandardScaler()`
  
  `y_train = sc_y.fit_transform(y_train)"""`

## Fitting Multiple Linear Regression to the Training set
  `from sklearn.linear_model import LinearRegression`
  
  `regressor = LinearRegression()`
  
  `regressor.fit(X_train, y_train)`

## Predicting the Test set results
  `y_pred = regressor.predict(X_test)`
  
# Backward Elimination Process Starts From Here
## Builind the optimal model using Backward Elimination

`import statsmodels.formula.api as sm`  // Library used for backword elimination 

`X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)`  // x0 = 1 Append Process

`X_opt = X[:, [0,1,2,3,4,5]]`   // Backward process starts to fit data according to p-value < sl

`regressor_OLS = sm.OLS(endog= y,exog= X_opt).fit()`

`regressor_OLS.summary()`

// Done this as index 2 has highest p value and it also greater than significant level
// So We have to remove that predictor

`X_opt = X[:, [0,1,3,4,5]]`

`regressor_OLS = sm.OLS(endog= y,exog= X_opt).fit()`

`regressor_OLS.summary()`

//Done this as index 1 has highest p value and it also greater than significant level
// So We have to remove that predictor

`X_opt = X[:, [0,3,4,5]]`

`regressor_OLS = sm.OLS(endog= y,exog= X_opt).fit()`

`regressor_OLS.summary()`

// Done this as index 4 has highest p value and it also greater than significant level
// So We have to remove that predictor

`X_opt = X[:, [0,3,5]]`

`regressor_OLS = sm.OLS(endog= y,exog= X_opt).fit()`

`regressor_OLS.summary()`

// Done this as index 5 has highest p value and it also greater than significant level
// So We have to remove that predictor

`X_opt = X[:, [0,3]]`

`regressor_OLS = sm.OLS(endog= y,exog= X_opt).fit()`

`regressor_OLS.summary()`
