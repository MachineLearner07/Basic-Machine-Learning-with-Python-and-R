# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Multiple Linear Regression to the Training set
#regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
#               data = training_set) is same as below 

regressor = lm(formula = Profit ~ .,
               data = training_set)

# summary(regressor)
 
# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Builind the optimal model using Backward Elimination with R

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset) 

summary(regressor)


# Will eliminate State as its corresponding p-value is higher than significant level


regressor = lm(formula = Profit ~ R.D.Spend   + Marketing.Spend ,
               data = dataset) 

summary(regressor)


# Will eliminate Administration as its corresponding p-value is higher than significant level

regressor = lm(formula = Profit ~ R.D.Spend  ,
               data = dataset) 

summary(regressor)


# Will eliminate Marketing.Spend as its corresponding p-value is higher than significant level

