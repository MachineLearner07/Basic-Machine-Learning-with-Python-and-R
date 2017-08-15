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
