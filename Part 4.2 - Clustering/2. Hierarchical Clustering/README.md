# 2. Hierarchical Clustering

![selection_184](https://user-images.githubusercontent.com/15044221/30268317-313c4ca0-9707-11e7-8fee-b2ea67a8402a.png)
![selection_185](https://user-images.githubusercontent.com/15044221/30268321-31527b4c-9707-11e7-8cee-c96f6e3e7bc4.png)
![selection_186](https://user-images.githubusercontent.com/15044221/30268320-31466532-9707-11e7-80a1-f115bcbfd465.png)
![selection_187](https://user-images.githubusercontent.com/15044221/30268318-3141803a-9707-11e7-86a0-7604a17f3766.png)
![selection_188](https://user-images.githubusercontent.com/15044221/30268319-314259ba-9707-11e7-9ae4-4d34f932f4ab.png)
![selection_189](https://user-images.githubusercontent.com/15044221/30268322-315b20a8-9707-11e7-942c-6fae4c4d0081.png)
![selection_190](https://user-images.githubusercontent.com/15044221/30268323-317df92a-9707-11e7-99c0-1fcfcea045aa.png)
![selection_191](https://user-images.githubusercontent.com/15044221/30268324-3189356a-9707-11e7-96ac-d90b4c084e49.png)
![selection_192](https://user-images.githubusercontent.com/15044221/30268325-31929d30-9707-11e7-8438-7c4aa0446b64.png)
![selection_193](https://user-images.githubusercontent.com/15044221/30268326-319ce452-9707-11e7-94c7-52feae443c40.png)
![selection_194](https://user-images.githubusercontent.com/15044221/30268327-31adf0da-9707-11e7-8a54-df7b3d9382bd.png)
![selection_195](https://user-images.githubusercontent.com/15044221/30268328-31f526bc-9707-11e7-8b4e-8630cf524063.png)
![selection_196](https://user-images.githubusercontent.com/15044221/30268329-322e25ac-9707-11e7-9d6b-4a0a188f5c99.png)
![selection_197](https://user-images.githubusercontent.com/15044221/30268332-32667286-9707-11e7-934a-376b8e72c030.png)
![selection_198](https://user-images.githubusercontent.com/15044221/30268331-325e7f5e-9707-11e7-8865-549563704be9.png)
![selection_199](https://user-images.githubusercontent.com/15044221/30268333-326c7cd0-9707-11e7-9aa4-40660fd36f6e.png)
![selection_200](https://user-images.githubusercontent.com/15044221/30268334-3298bebc-9707-11e7-8a9c-45f99e20524e.png)
![selection_201](https://user-images.githubusercontent.com/15044221/30268335-32b885ee-9707-11e7-9ff5-98f81a6cee08.png)
![selection_202](https://user-images.githubusercontent.com/15044221/30268336-32dec510-9707-11e7-9665-482969f43eaf.png)
![selection_203](https://user-images.githubusercontent.com/15044221/30268337-32fca7b0-9707-11e7-8cef-aca227e708d3.png)
![selection_204](https://user-images.githubusercontent.com/15044221/30268339-332401e8-9707-11e7-9124-34d6a6a1843f.png)
![selection_205](https://user-images.githubusercontent.com/15044221/30268340-333268be-9707-11e7-9a5f-31df40849717.png)
![selection_206](https://user-images.githubusercontent.com/15044221/30268342-33456400-9707-11e7-8180-ab44200fd1df.png)
![selection_207](https://user-images.githubusercontent.com/15044221/30268343-335184b0-9707-11e7-9cbd-d3bd6b829d4f.png)
![selection_208](https://user-images.githubusercontent.com/15044221/30268344-335f5eb4-9707-11e7-935f-b69172951cf8.png)
![selection_209](https://user-images.githubusercontent.com/15044221/30268345-33763dfa-9707-11e7-93a1-1ea5280c0b95.png)
![selection_210](https://user-images.githubusercontent.com/15044221/30268347-339ce978-9707-11e7-94c3-6b94e035bfcd.png)
![selection_211](https://user-images.githubusercontent.com/15044221/30268346-33929d2e-9707-11e7-86e7-3cd59af8f796.png)


**Hierarchical Clustering Code**

```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the mall dataset with pansdas
dataset = pd.read_csv('mall.csv')
X = dataset.iloc[:, [3,4]].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
```

![selection_212](https://user-images.githubusercontent.com/15044221/30268348-33a49a1a-9707-11e7-83b7-b6593f749d36.png)

```python
#Fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity ='euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0,1], s = 100, c = 'red' , label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1,1], s = 100, c = 'blue' , label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2,1], s = 100, c = 'green' , label = 'Traget')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3,1], s = 100, c = 'cyan' , label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4,1], s = 100, c = 'magenta' , label = 'Sensible')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```

![selection_214](https://user-images.githubusercontent.com/15044221/30268349-33af69a4-9707-11e7-8d89-c32f9af7555a.png)
