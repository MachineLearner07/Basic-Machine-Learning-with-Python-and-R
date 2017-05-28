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
  ##### 
  
 
