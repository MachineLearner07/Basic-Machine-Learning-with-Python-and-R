  ### 1. Logistic Regression
  --------------------------
  ![selection_025](https://user-images.githubusercontent.com/15044221/29221315-221e025c-7ee0-11e7-9008-d04ccad5d755.png)
  ![selection_026](https://user-images.githubusercontent.com/15044221/29221326-2be6623e-7ee0-11e7-95be-03d75f90bd7c.png)
  ![selection_027](https://user-images.githubusercontent.com/15044221/29221330-2edac9c6-7ee0-11e7-93cb-8becabe854ae.png)

You will find `Social_Network_Ads.csv` <a href="https://github.com/MachineLearner07/Basic-Machine-Learning-with-Python-and-R/blob/rezwan/Part%203.2%20-%20Classsification/1.%20Logistic%20Regression/Social_Network_Ads.csv"> here</a>

**Logistic Regression code**
```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values # Age and EstimatedSalary
y = dataset.iloc[:, 4].values # Purchased

# Spliting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```
- Confusion Matrix
![selection_120](https://user-images.githubusercontent.com/15044221/29446452-14460b7e-840e-11e7-9f1c-e8e48ed01115.png)

* Visualising of Training set results  

```python
# Visualising of Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regresssion (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

![selection_029](https://user-images.githubusercontent.com/15044221/29251324-a359a7c8-8074-11e7-8717-0b6cb03e3d51.png)


* Visualising of Test set results

```python
# Visualising of Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regresssion (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

![selection_030](https://user-images.githubusercontent.com/15044221/29251391-a2b516d0-8075-11e7-8e0b-f32a520e3697.png)

