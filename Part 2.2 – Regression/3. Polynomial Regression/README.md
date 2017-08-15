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
