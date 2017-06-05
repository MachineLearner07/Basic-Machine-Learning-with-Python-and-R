# Basic-Machine-Learning-with-Python-and-R

## What is Machine Learning ?
##### According to Tom Michel : 
	A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.
	
##### According to Arthur Samuel : 
	Field of study that gives computers the ability to learn without being explicitely programmed.
	
## Part 1.2 - Data Preprocessing 

Data pre-processing is an important step in the data mining process. The phrase"garbage in, garbage out"is particularly applicable to data mining andmachine learningprojects. Data-gathering methods are often loosely controlled, resulting inout-of-rangevalues (e.g., Income: ?100), impossible data combinations (e.g., Sex: Male, Pregnant: Yes),missing values, etc. Analyzing data that has not been carefully screened for such problems can produce misleading results. Thus, the representation andquality of datais first and foremost before running an analysis. 

If there is much irrelevant and redundant information present or noisy and unreliable data, thenknowledge discoveryduring the training phase is more difficult. Data preparation and filtering steps can take considerable amount of processing time. Data pre-processing includescleaning,Instance selection,normalization,transformation,feature extractionandselection, etc. The product of data pre-processing is the finaltraining set. Kotsiantis et al. (2006) present a well-known algorithm for each step of data pre-processing. 

```python
# Data Preprocessing 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
```
#### Dataset : 
![selection_013](https://cloud.githubusercontent.com/assets/15044221/26520233/428819fa-42f0-11e7-8c1b-b531daccda4b.png)
```python
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#### Missing values : X ->
array([['France', 44.0, 72000.0],
       ['Spain', 27.0, 48000.0],
       ['Germany', 30.0, 54000.0],
       ['Spain', 38.0, 61000.0],
       ['Germany', 40.0, 63777.77777777778],
       ['France', 35.0, 58000.0],
       ['Spain', 38.77777777777778, 52000.0],
       ['France', 48.0, 79000.0],
       ['Germany', 50.0, 83000.0],
       ['France', 37.0, 67000.0]], dtype=object)
#### Missing values : y -> 
 array(['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes'], dtype=object)
```
```python
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
```

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
   ##### 
   Simple linear regression is a statistical method that allows us to summarize and study relationships between two continuous (quantitative) variables. 
   
![selection_005](https://cloud.githubusercontent.com/assets/15044221/26520188/934432bc-42ef-11e7-847c-5f4c2c66d945.png)
![selection_004](https://cloud.githubusercontent.com/assets/15044221/26520189/965d2a12-42ef-11e7-894f-5461c5292c2f.png)
![selection_003](https://cloud.githubusercontent.com/assets/15044221/26520191/9c00697a-42ef-11e7-8547-aa20a223f7a3.png)

```python
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
#### Dataset : 
![selection_006](https://cloud.githubusercontent.com/assets/15044221/26520095/d50c5456-42ed-11e7-8ac3-f6cfb52664a3.png)
![selection_007](https://cloud.githubusercontent.com/assets/15044221/26520096/d8ca863a-42ed-11e7-8a65-6746d5ed9476.png)
#### X_train, X_test, y_train, y_test : 
![selection_008](https://cloud.githubusercontent.com/assets/15044221/26520097/e0e9e720-42ed-11e7-9b74-ca562c7ef0a3.png)
![selection_009](https://cloud.githubusercontent.com/assets/15044221/26520099/e2311e8c-42ed-11e7-8533-ea8ee47a7915.png)
```python
# Fitting simple linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test results
y_pred = regressor.predict(X_test)
```
#### y_pred : 
![selection_010](https://cloud.githubusercontent.com/assets/15044221/26520100/e5d84c4a-42ed-11e7-8cf0-0b817fc5361f.png)
```python
# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salay')
plt.show()
```
#### Visualising the Training set results
![selection_011](https://cloud.githubusercontent.com/assets/15044221/26520101/e87ebfe2-42ed-11e7-8038-42ed3730767d.png)
N.B : blue color = Real values, red color = Predicting values
```python
# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salay')
plt.show()
```
#### Visualising the Test set results
![selection_012](https://cloud.githubusercontent.com/assets/15044221/26520103/ed451ac6-42ed-11e7-9b06-9e38fec899ad.png)
N.B : blue color = Real values, red color = Predicting values

 ### 2. Multiple Linear Regression 
  ##### Def:
  	Multiple linear regression (MLR) is a statistical technique that uses several explanatory variables to predict the outcome of a response variable. The goal of multiple linear regression (MLR) is to model the relationship between the explanatory and response variables.

The model for MLR, given n observations, is:

yi = B0 + B1xi1 + B2xi2 + ... + Bpxip + Ei where i = 1,2, ..., n

![selection_016](https://cloud.githubusercontent.com/assets/15044221/26530965/addc568c-4401-11e7-8aa5-e0c31dea5e04.png)
![selection_017](https://cloud.githubusercontent.com/assets/15044221/26530967/b0d96ee2-4401-11e7-93e4-c52121487e0b.png)
![selection_018](https://cloud.githubusercontent.com/assets/15044221/26530968/b36abff8-4401-11e7-85b5-5536441d62d4.png)

```python
# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

###### X : 
array([[165349.2, 136897.8, 471784.1, 'New York'],
       [162597.7, 151377.59, 443898.53, 'California'],
       [153441.51, 101145.55, 407934.54, 'Florida'],
       [144372.41, 118671.85, 383199.62, 'New York'],
       [142107.34, 91391.77, 366168.42, 'Florida'],
       [131876.9, 99814.71, 362861.36, 'New York'],
       [134615.46, 147198.87, 127716.82, 'California'],
       [130298.13, 145530.06, 323876.68, 'Florida'],
       [120542.52, 148718.95, 311613.29, 'New York'],
       [123334.88, 108679.17, 304981.62, 'California'],
       [101913.08, 110594.11, 229160.95, 'Florida'],
       [100671.96, 91790.61, 249744.55, 'California'],
       [93863.75, 127320.38, 249839.44, 'Florida'],
       [91992.39, 135495.07, 252664.93, 'California'],
       [119943.24, 156547.42, 256512.92, 'Florida'],
       [114523.61, 122616.84, 261776.23, 'New York'],
       [78013.11, 121597.55, 264346.06, 'California'],
       [94657.16, 145077.58, 282574.31, 'New York'],
       [91749.16, 114175.79, 294919.57, 'Florida'],
       [86419.7, 153514.11, 0.0, 'New York'],
       [76253.86, 113867.3, 298664.47, 'California'],
       [78389.47, 153773.43, 299737.29, 'New York'],
       [73994.56, 122782.75, 303319.26, 'Florida'],
       [67532.53, 105751.03, 304768.73, 'Florida'],
       [77044.01, 99281.34, 140574.81, 'New York'],
       [64664.71, 139553.16, 137962.62, 'California'],
       [75328.87, 144135.98, 134050.07, 'Florida'],
       [72107.6, 127864.55, 353183.81, 'New York'],
       [66051.52, 182645.56, 118148.2, 'Florida'],
       [65605.48, 153032.06, 107138.38, 'New York'],
       [61994.48, 115641.28, 91131.24, 'Florida'],
       [61136.38, 152701.92, 88218.23, 'New York'],
       [63408.86, 129219.61, 46085.25, 'California'],
       [55493.95, 103057.49, 214634.81, 'Florida'],
       [46426.07, 157693.92, 210797.67, 'California'],
       [46014.02, 85047.44, 205517.64, 'New York'],
       [28663.76, 127056.21, 201126.82, 'Florida'],
       [44069.95, 51283.14, 197029.42, 'California'],
       [20229.59, 65947.93, 185265.1, 'New York'],
       [38558.51, 82982.09, 174999.3, 'California'],
       [28754.33, 118546.05, 172795.67, 'California'],
       [27892.92, 84710.77, 164470.71, 'Florida'],
       [23640.93, 96189.63, 148001.11, 'California'],
       [15505.73, 127382.3, 35534.17, 'New York'],
       [22177.74, 154806.14, 28334.72, 'California'],
       [1000.23, 124153.04, 1903.93, 'New York'],
       [1315.46, 115816.21, 297114.46, 'Florida'],
       [0.0, 135426.92, 0.0, 'California'],
       [542.05, 51743.15, 0.0, 'New York'],
       [0.0, 116983.8, 45173.06, 'California']], dtype=object)
```
```python
# Encoding categorical data
# Endcoding the Independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform( X[:, 3] )
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
```
##### X:
![selection_020](https://cloud.githubusercontent.com/assets/15044221/26590542/b0366b8a-457c-11e7-9b8c-2d2effc1e627.png)

```python
# Avoiding the Dummy variable trap
X = X[:, 1:]
```
##### X:
![selection_021](https://cloud.githubusercontent.com/assets/15044221/26590671/04bc5066-457d-11e7-9b78-70e15cb09eaa.png)

```python
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression for the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set result
y_pred = regressor.predict(X_test)
```
##### y_pred:
![selection_022](https://cloud.githubusercontent.com/assets/15044221/26590778/5bf98bd2-457d-11e7-88f6-32462d1bb1c2.png)

##### Backword Elimination
![selection_023](https://cloud.githubusercontent.com/assets/15044221/26626536/56bf4a5c-4619-11e7-8dcf-cf5056da4402.png)
```python
# Building the optimal model using Backword Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int) , values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
```

### 3. Polynomial Regression
###### Def:
	In statistics, polynomial regression is a form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modelled as an nth degree polynomial in x.
![325px-polyreg_scheffe svg](https://cloud.githubusercontent.com/assets/15044221/26672705/f84bc39c-46db-11e7-91db-16a12a54d233.png)

![selection_024](https://cloud.githubusercontent.com/assets/15044221/26673086/7cc95fac-46dd-11e7-9f5a-d28b475b20b3.png)

```python
# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv') 
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
```
![capture2](https://cloud.githubusercontent.com/assets/15044221/26760028/85d984a0-4930-11e7-81da-35120703ea0f.PNG)
![capture](https://cloud.githubusercontent.com/assets/15044221/26760006/e67113c4-492f-11e7-9414-a58854630ba7.PNG)

```python
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
```
![capture1](https://cloud.githubusercontent.com/assets/15044221/26760031/89100bb2-4930-11e7-9191-c55513c1e81c.PNG)
##### Visualising the Linear Regression
```python
# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```
![capture3](https://cloud.githubusercontent.com/assets/15044221/26760056/0e490a0e-4931-11e7-9054-04444fc6bbf6.PNG)
##### Visualising the Polynomial Regression
```python
# Visualising the Polynomial Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```
![capture4](https://cloud.githubusercontent.com/assets/15044221/26760070/6be17e6c-4931-11e7-937b-499ad85c5802.PNG)
##### Predicting a New result
```python
# Predicting a new results with Linear Regression
lin_reg.predict(6.5) 

Out[35]: array([ 330378.78787879])

# Predicting a new results with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))

Out[36]: array([ 158862.45265153])

```
### 4. Support Vector for Regression (SVR)
	SVM, which stands for Support Vector Machine, is a classifier. Classifiers perform classification, predicting discrete categorical 	labels. SVR, which stands for Support Vector Regressor, is a regressor. Regressors perform regression, predicting continuous ordered variables. Both use very similar algorithms, but predict different types of variables.

```python
# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv') 
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into Training set and Test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)'''

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting the Regression Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```
![capture5](https://cloud.githubusercontent.com/assets/15044221/26760682/e7eda332-4940-11e7-8ec4-c182c4d7900f.PNG)
```python
# Visualising the SVR results (for higher resolation and smoother results)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```
![capture6](https://cloud.githubusercontent.com/assets/15044221/26760684/ea57217a-4940-11e7-991f-0e6faab6105d.PNG)

### 5. Decision Tree Regresion
	Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. In decision analysis, a decision tree can be used to visually and explicitly represent decisions and decision making.
![capture7](https://cloud.githubusercontent.com/assets/15044221/26777719/0e06262e-4a00-11e7-99dc-7d87169583c2.PNG)
```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv') 
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting the Decision Tree Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

# Predicting a new result
y_pred = regressor.predict(6.5)
# Output : y_pred = 1.50e+05

# Visualising the Decision Tree Regression results (for higher resolation and smoother results)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```
![capture8](https://cloud.githubusercontent.com/assets/15044221/26777874/b94ef128-4a00-11e7-8d63-da04e202b93e.PNG)

