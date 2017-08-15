## Part 2  Regression

Regression models (both linear and non-linear) are used for predicting a real value, like salary for example. If yourindependent variable is time, then you are forecasting future values, otherwise your model is predicting present butunknown values. Regression technique vary from Linear Regression to SVR and RandomForests Regression.
In this part, you will understand and learn how to implement the following Machine Learning Regression models:

 - [1. Simple Linear Regression](https://github.com/MachineLearner07/Basic-Machine-Learning-with-Python-and-R/tree/rezwan/Part%202.2%20%E2%80%93%20Regression/1.%20Simple%20Linear%20Regression)
 - [2. Multiple Linear Regression](https://github.com/MachineLearner07/Basic-Machine-Learning-with-Python-and-R/tree/rezwan/Part%202.2%20%E2%80%93%20Regression/2.%20Multiple%20Linear%20Regression)
 - [3. Polynomial Regression](https://github.com/MachineLearner07/Basic-Machine-Learning-with-Python-and-R/tree/rezwan/Part%202.2%20%E2%80%93%20Regression/3.%20Polynomial%20Regression)
 - [4. Support Vector for Regression (SVR)](https://github.com/MachineLearner07/Basic-Machine-Learning-with-Python-and-R/tree/rezwan/Part%202.2%20%E2%80%93%20Regression/4.%20Support%20Vector%20for%20Regression%20(SVR))
 - [5. Decision Tree Classification](https://github.com/MachineLearner07/Basic-Machine-Learning-with-Python-and-R/tree/rezwan/Part%202.2%20%E2%80%93%20Regression/5.%20Decision%20Tree%20Regression)
 - [6. Random Forest Classification](https://github.com/MachineLearner07/Basic-Machine-Learning-with-Python-and-R/tree/rezwan/Part%202.2%20%E2%80%93%20Regression/6.%20Random%20Forest%20Regression)

#### Conclusion of Part 2 - Regression

After learning about these six regression models, you are probably asking yourself the following questions:

What are the pros and cons of each model ?
How do I know which model to choose for my problem ?
How can I improve each of these models ?
Let's answer each of these questions one by one:

1. What are the pros and cons of each model ?

Please find  <a href="http://www.superdatascience.com/wp-content/uploads/2017/02/Regression-Pros-Cons.pdf">here</a> a cheat-sheet that gives you all the pros and the cons of each regression model.

2. How do I know which model to choose for my problem ?

First, you need to figure out whether your problem is linear or non linear. You will learn how to do that in Part 10 - Model Selection. Then:

If your problem is linear, you should go for Simple Linear Regression if you only have one feature, and Multiple Linear Regression if you have several features.

If your problem is non linear, you should go for Polynomial Regression, SVR, Decision Tree or Random Forest. Then which one should you choose among these four ? That you will learn in Part 10 - Model Selection. The method consists of using a very relevant technique that evaluates your models performance, called k-Fold Cross Validation, and then picking the model that shows the best results. Feel free to jump directly to Part 10 if you already want to learn how to do that.

3. How can I improve each of these models ?

In Part 10 - Model Selection, you will find the second section dedicated to Parameter Tuning, that will allow you to improve the performance of your models, by tuning them. You probably already noticed that each model is composed of two types of parameters:

the parameters that are learnt, for example the coefficients in Linear Regression,
the hyperparameters.
The hyperparameters are the parameters that are not learnt and that are fixed values inside the model equations. For example, the regularization parameter lambda or the penalty parameter C are hyperparameters. So far we used the default value of these hyperparameters, and we haven't searched for their optimal value so that your model reaches even higher performance. Finding their optimal value is exactly what Parameter Tuning is about. So for those of you already interested in improving your model performance and doing some parameter tuning, feel free to jump directly to Part 10 - Model Selection.

And as a BONUS, please find <a href="http://www.superdatascience.com/wp-content/uploads/2017/02/Regularization.pdf">here</a> some slides we made about Regularization.

----------------------------------------------------------------------------------------------------------------------------

