# Basic-Machine-Learning-with-Python-and-R

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

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

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
   ![screenshot from 2017-05-27 00-07-36](https://cloud.githubusercontent.com/assets/15044221/26509615/56b55336-427b-11e7-9a25-4dfb295a7a08.png)
   ![screenshot from 2017-05-27 00-11-50](https://cloud.githubusercontent.com/assets/15044221/26509717/cc5ccae2-427b-11e7-94f6-6675965fc4c0.png)
   ![4](https://cloud.githubusercontent.com/assets/15044221/26510293/708f2266-427e-11e7-88df-9811b75ec53c.png)

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

dataset
Out[7]: 
    YearsExperience    Salary
0               1.1   39343.0
1               1.3   46205.0
2               1.5   37731.0
3               2.0   43525.0
4               2.2   39891.0
5               2.9   56642.0
6               3.0   60150.0
7               3.2   54445.0
8               3.2   64445.0
9               3.7   57189.0
10              3.9   63218.0
11              4.0   55794.0
12              4.0   56957.0
13              4.1   57081.0
14              4.5   61111.0
15              4.9   67938.0
16              5.1   66029.0
17              5.3   83088.0
18              5.9   81363.0
19              6.0   93940.0
20              6.8   91738.0
21              7.1   98273.0
22              7.9  101302.0
23              8.2  113812.0
24              8.7  109431.0
25              9.0  105582.0
26              9.5  116969.0
27              9.6  112635.0
28             10.3  122391.0
29             10.5  121872.0

X_train
Out[8]: 
array([[  2.9],
       [  5.1],
       [  3.2],
       [  4.5],
       [  8.2],
       [  6.8],
       [  1.3],
       [ 10.5],
       [  3. ],
       [  2.2],
       [  5.9],
       [  6. ],
       [  3.7],
       [  3.2],
       [  9. ],
       [  2. ],
       [  1.1],
       [  7.1],
       [  4.9],
       [  4. ]])

X_test
Out[9]: 
array([[  1.5],
       [ 10.3],
       [  4.1],
       [  3.9],
       [  9.5],
       [  8.7],
       [  9.6],
       [  4. ],
       [  5.3],
       [  7.9]])

y_train
Out[10]: 
array([  56642.,   66029.,   64445.,   61111.,  113812.,   91738.,
         46205.,  121872.,   60150.,   39891.,   81363.,   93940.,
         57189.,   54445.,  105582.,   43525.,   39343.,   98273.,
         67938.,   56957.])

y_test
Out[11]: 
array([  37731.,  122391.,   57081.,   63218.,  116969.,  109431.,
        112635.,   55794.,   83088.,  101302.])
 
```

