# R Data Preprocessing Template
  
  ##### Importing the dataset
  
  //Before importing dataset set working directory
  
  `dataset = read.csv('Salary_Data.csv')`

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


# Simple Linear Regression with R

##### Fitting Simple Linear Regression to the Training set

// lm is used to fit linear models. 

//regressor is an object name

`regressor = lm(formula = Salary ~ YearsExperience,data = training_set)`  

##### Predicting the Test set results
`y_pred = predict(regressor, newdata = test_set)`

##### Visualising the Training set results
  `#install.packages("ggplot2")`
  
  `library(ggplot2)`
  `ggplot() +`
  
  `geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),colour = 'red') +`
  
  `geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),colour = 'blue') +`
  
  `ggtitle('Salary vs Experience (Training set)') +`
  
  `xlab('Years of experience') +`
  
  `ylab('Salary')`

##### Visualising the Test set results

  `library(ggplot2)`
  
  `ggplot() +`
  
    `geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), colour = 'red') +`
    
    `geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), colour = 'blue') +`
    
    `ggtitle('Salary vs Experience (Test set)') +`
    
    `xlab('Years of experience') +`
    
    `ylab('Salary')`
