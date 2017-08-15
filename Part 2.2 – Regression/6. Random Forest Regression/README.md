### 6. Random Forest Regresion
![capture9](https://cloud.githubusercontent.com/assets/15044221/26793070/6d23aa9e-4a3e-11e7-8515-e868ae981288.PNG)

```python
# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting the Random Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)
# y_pred = 1.603333333333333430e+05

# Visualising the Random Forest Regression results (for higher resolation and smoother results)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```
![capture10](https://cloud.githubusercontent.com/assets/15044221/26793531/1b88de6e-4a40-11e7-94b6-5849aa60b862.PNG)
