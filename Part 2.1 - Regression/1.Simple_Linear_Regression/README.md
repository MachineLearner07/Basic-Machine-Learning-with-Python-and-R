# R Data Preprocessing Template
  
  ##### Importing the dataset
  
  //Before importing dataset set working directory
  
  `dataset = read.csv('Data.csv')`

  ##### Splitting the dataset into the Training set and Test set
  `#install.packages('caTools')` //Use it if 'caTools' is not installed
  
  `library(caTools)`  // Include Library
  
  `set.seed(123)`  //Have to set seed every time when we want to get a reproducible random result
  
  `split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)`  //Use For Spliting data for test set and training set
  
  `training_set = subset(dataset, split == TRUE)`
  
  `test_set = subset(dataset, split == FALSE)`

  ##### Feature Scaling
  // Feature scaling is a method used to standardize the range of independent variables or features of data. Feature Scaling is not always necessary in data preprocessing steps
  
  `#training_set = scale(training_set)`
  
  `#test_set = scale(test_set)`
