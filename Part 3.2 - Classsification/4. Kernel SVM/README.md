## 4. Kernel SVM

![selection_052](https://user-images.githubusercontent.com/15044221/29360268-fef39bb2-82a3-11e7-999f-e0f353ac1c1d.png)
![selection_053](https://user-images.githubusercontent.com/15044221/29360270-fef76616-82a3-11e7-8b1b-b01bc8fd5e6e.png)
![selection_054](https://user-images.githubusercontent.com/15044221/29360273-fef99472-82a3-11e7-9448-aa3f9e729dde.png)
![selection_055](https://user-images.githubusercontent.com/15044221/29360271-fef7ddc6-82a3-11e7-9a71-b83d70e5365f.png)
![selection_056](https://user-images.githubusercontent.com/15044221/29360267-fef32506-82a3-11e7-8545-2c008292565f.png)
![selection_057](https://user-images.githubusercontent.com/15044221/29360269-fef75c34-82a3-11e7-925d-cd725c1ae9e6.png)
![selection_058](https://user-images.githubusercontent.com/15044221/29360274-ff2ee97e-82a3-11e7-969c-30b7d0a3d26c.png)
![selection_059](https://user-images.githubusercontent.com/15044221/29360276-ff363c24-82a3-11e7-89af-e648d975dae1.png)
![selection_060](https://user-images.githubusercontent.com/15044221/29360275-ff35d996-82a3-11e7-8034-5df4312eb334.png)
![selection_061](https://user-images.githubusercontent.com/15044221/29360277-ff380c52-82a3-11e7-815f-954495bdb1d1.png)
![selection_062](https://user-images.githubusercontent.com/15044221/29360278-ff3c2c92-82a3-11e7-937f-0bd13392bd05.png)
![selection_063](https://user-images.githubusercontent.com/15044221/29360279-ff538af4-82a3-11e7-8b8a-5e356321f838.png)
![selection_064](https://user-images.githubusercontent.com/15044221/29360280-ff67b7f4-82a3-11e7-9318-25d74babbbe3.png)
![selection_065](https://user-images.githubusercontent.com/15044221/29360281-ff77bb7c-82a3-11e7-8733-a45874dcdf79.png)
![selection_066](https://user-images.githubusercontent.com/15044221/29360282-ff796030-82a3-11e7-984a-ebe31b7d7c38.png)


### Kernel SVM Code
```python
# Kernel SVM

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

# Fitting the kernel classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

- Visualising the Training set results of Kernel SVM
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
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

![selection_050](https://user-images.githubusercontent.com/15044221/29360479-ea2de808-82a4-11e7-82c1-f98016e3e120.png)

- Visualising the Test set results of Kernel SVM
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
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

```

![selection_051](https://user-images.githubusercontent.com/15044221/29360482-ee9957d8-82a4-11e7-9a5e-a677e84e4534.png)

