Clustering is similar to classification, but the basis is different. In Clustering you don’t know what you are looking for, and you are trying to identify some segments or clusters in your data. When you use clustering algorithms on your dataset, unexpected things can suddenly pop up like structures, clusters and groupings you would have never thought of otherwise.

### K - Means Clustering

`Work flow of K-means Clustering`

![image](https://user-images.githubusercontent.com/20562497/31102664-ef0a029c-a7f4-11e7-8433-0a73331a8728.png)

### Random Initialization Trap of K-Means clustering

Original Classification
![image](https://user-images.githubusercontent.com/20562497/31103296-face8d70-a7f7-11e7-8d48-3117ffd0e821.png)

Wrong Classification
![image](https://user-images.githubusercontent.com/20562497/31103341-2fd66dc6-a7f8-11e7-8ba5-b3ff61d1edf6.png)

###### Solution of Random Initialization Trap of K-Means clustering is K-Means++

### Algorithm behinds Choosing the right number of clusters

![image](https://user-images.githubusercontent.com/20562497/31104261-d8488822-a7fd-11e7-9842-28b5a0b0d4fd.png)

` N.B : The lesser the WCSS value, Higher the cluster, Data points Also fit Better.`

![image](https://user-images.githubusercontent.com/20562497/31104519-385a952e-a7ff-11e7-842b-92fcef643450.png)
