## Part 1 - Data Preprocessing

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

----------------------------------------------------------------------------------------------------------------------------
