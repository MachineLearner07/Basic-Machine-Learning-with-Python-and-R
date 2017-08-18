# 6. Decision Tree Classification

![selection_121](https://user-images.githubusercontent.com/15044221/29456245-ab81f318-8436-11e7-97e3-98814597be5c.png)
![selection_122](https://user-images.githubusercontent.com/15044221/29456244-ab7e1c16-8436-11e7-8616-97d0e70920e7.png)
![selection_123](https://user-images.githubusercontent.com/15044221/29456246-abf6d106-8436-11e7-8516-2ac1e0e97646.png)
![selection_124](https://user-images.githubusercontent.com/15044221/29456248-ac0e7a68-8436-11e7-917b-9b092929752c.png)
![selection_125](https://user-images.githubusercontent.com/15044221/29456247-ac089288-8436-11e7-8b18-a1bfc3a69271.png)
![selection_126](https://user-images.githubusercontent.com/15044221/29456251-accd580c-8436-11e7-9a6a-08b8d81a6140.png)
![selection_127](https://user-images.githubusercontent.com/15044221/29456249-ac2f05ee-8436-11e7-8264-be7414f76dda.png)
![selection_128](https://user-images.githubusercontent.com/15044221/29456250-ac573a64-8436-11e7-94b7-5ecc1e547058.png)

- Decision Tree Classification Code
```python
# Decision Tree Regression

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

# Fitting the classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```
- Confusion Matrix
![selection_129](https://user-images.githubusercontent.com/15044221/29456337-068c1da6-8437-11e7-8d98-329b58588aff.png)

- Visualising of Training set results of Decision Tree Classifier
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
plt.title('Decision Tree (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

```
![selection_130](https://user-images.githubusercontent.com/15044221/29456358-19ac773c-8437-11e7-8407-73128e856b84.png)

- Visualising of Test set results of Decision Tree Classifier
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
plt.title('Decision Tree (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```
![selection_131](https://user-images.githubusercontent.com/15044221/29456357-19aafc2c-8437-11e7-9822-7158ae028714.png)
