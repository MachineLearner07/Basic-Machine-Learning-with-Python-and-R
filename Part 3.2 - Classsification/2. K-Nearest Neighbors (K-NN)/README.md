  ### 2. K-Nearest Neighbors (K-NN)
  
![selection_031](https://user-images.githubusercontent.com/15044221/29266909-3fdba216-8108-11e7-8df5-31c6ec4a3f95.png)
![selection_032](https://user-images.githubusercontent.com/15044221/29266913-44240908-8108-11e7-9857-138780e9b612.png)
![selection_034](https://user-images.githubusercontent.com/15044221/29266922-4d825fd6-8108-11e7-9633-6c462cd78ca0.png)
![selection_033](https://user-images.githubusercontent.com/15044221/29266933-5919de0a-8108-11e7-8f84-f2408c30ec0d.png)
![selection_035](https://user-images.githubusercontent.com/15044221/29266938-5db397b2-8108-11e7-838b-24ee37b2462f.png)
![selection_036](https://user-images.githubusercontent.com/15044221/29266945-62ed8c24-8108-11e7-8be0-4d5e4c5ad847.png)
![selection_037](https://user-images.githubusercontent.com/15044221/29266952-67975a3e-8108-11e7-9e2e-a164dc555bbb.png)


**K-Nearest Neighbors (K-NN) code**
```python
# K-Nearest Neighbors (K-NN)

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
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```
- Confusion Matrix
![selection_119](https://user-images.githubusercontent.com/15044221/29446256-08eaf2ea-840d-11e7-8e80-3bc2b78d4d7c.png)

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
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```
![selection_038](https://user-images.githubusercontent.com/15044221/29279920-31370ee8-813b-11e7-96f7-c235c09388e0.png)

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
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```
![selection_039](https://user-images.githubusercontent.com/15044221/29279923-3318012c-813b-11e7-9cb8-f9069488efc5.png)
