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
