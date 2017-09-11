# 1. K-Means Clustering

![selection_151](https://user-images.githubusercontent.com/15044221/30259569-e725aa08-96e3-11e7-9c3f-ed7fb372de47.png)
![selection_152](https://user-images.githubusercontent.com/15044221/30259573-e764abae-96e3-11e7-8d0c-3bbb33cc71da.png)
![selection_153](https://user-images.githubusercontent.com/15044221/30259574-e78e0846-96e3-11e7-8db4-7a7e032d3ea6.png)
![selection_154](https://user-images.githubusercontent.com/15044221/30259575-e78eb142-96e3-11e7-8d4e-aed0a0cd0d12.png)
![selection_155](https://user-images.githubusercontent.com/15044221/30259576-e78f1844-96e3-11e7-8029-6dee9367dd23.png)
![selection_156](https://user-images.githubusercontent.com/15044221/30259577-e7912fa8-96e3-11e7-8acd-895c91f16a6d.png)
![selection_157](https://user-images.githubusercontent.com/15044221/30259578-e792243a-96e3-11e7-98ff-ea644b6e3f46.png)
![selection_158](https://user-images.githubusercontent.com/15044221/30259579-e7a2ca92-96e3-11e7-9344-e55e772e5bfd.png)
![selection_159](https://user-images.githubusercontent.com/15044221/30259581-e7c90446-96e3-11e7-8a4a-66ab7a2063ac.png)
![selection_160](https://user-images.githubusercontent.com/15044221/30259582-e7cb9012-96e3-11e7-8de4-cd53956dca0d.png)
![selection_161](https://user-images.githubusercontent.com/15044221/30259583-e7ce568a-96e3-11e7-83a2-e1dce53be2f2.png)
![selection_162](https://user-images.githubusercontent.com/15044221/30259584-e7cfe464-96e3-11e7-9106-11f3937503c8.png)
![selection_163](https://user-images.githubusercontent.com/15044221/30259585-e7d276ac-96e3-11e7-9203-e9512d028b0d.png)
![selection_164](https://user-images.githubusercontent.com/15044221/30259586-e7e2603a-96e3-11e7-9cac-01881f08b6ff.png)
![selection_165](https://user-images.githubusercontent.com/15044221/30259587-e803d6f2-96e3-11e7-9dfc-3c63f6cec12f.png)
![selection_166](https://user-images.githubusercontent.com/15044221/30259588-e807f0c0-96e3-11e7-9a39-caba1327f832.png)
![selection_167](https://user-images.githubusercontent.com/15044221/30259589-e8099a06-96e3-11e7-9b77-6ff60367618d.png)
![selection_168](https://user-images.githubusercontent.com/15044221/30259590-e810402c-96e3-11e7-93a4-a4e395a748fc.png)
![selection_169](https://user-images.githubusercontent.com/15044221/30259591-e8147b4c-96e3-11e7-89ed-d5fadf6fce1f.png)
![selection_170](https://user-images.githubusercontent.com/15044221/30259592-e820cc30-96e3-11e7-8a3a-b609c9323249.png)
![selection_171](https://user-images.githubusercontent.com/15044221/30259593-e83e7bfe-96e3-11e7-8c6e-12a6c2339265.png)
![selection_172](https://user-images.githubusercontent.com/15044221/30259594-e843a1a6-96e3-11e7-8bd5-d1fe385ebe51.png)
![selection_173](https://user-images.githubusercontent.com/15044221/30259595-e84682fe-96e3-11e7-830c-b7850b1c82e4.png)

You will find `Mall_Customers.csv` <a href="https://github.com/MachineLearner07/Basic-Machine-Learning-with-Python-and-R/blob/rezwan/Part%204.2%20-%20Clustering/1.%20%20K-Means%20Clustering/Mall_Customers.csv"> here</a>

**K-Means Clustering code**
```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Mall_Customers dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# Using the Elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```

**Curve Elbow method**

![selection_182](https://user-images.githubusercontent.com/15044221/30259840-65e9218e-96e5-11e7-8a8e-1620ce011218.png)

```python
# Applying k-means to the mall dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_means == 0, 0], X[y_means == 0,1], s = 100, c = 'red' , label = 'Careful')
plt.scatter(X[y_means == 1, 0], X[y_means == 1,1], s = 100, c = 'blue' , label = 'Standard')
plt.scatter(X[y_means == 2, 0], X[y_means == 2,1], s = 100, c = 'green' , label = 'Traget')
plt.scatter(X[y_means == 3, 0], X[y_means == 3,1], s = 100, c = 'cyan' , label = 'Careless')
plt.scatter(X[y_means == 4, 0], X[y_means == 4,1], s = 100, c = 'magenta' , label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```

**Visualizing of Dataset**

![selection_183](https://user-images.githubusercontent.com/15044221/30259841-66040b16-96e5-11e7-91e4-5a7cef89b194.png)

