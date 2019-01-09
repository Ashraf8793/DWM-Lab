import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#import the dataset ( like csv file or any format file)
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,:1].values

#spliliting the data set inti trainig and test set
from sklearn.model_selection import
train_test_split

#used model_selection in place of cross_validation since the latter is seprecated_
X_train, X_test, y_train, y_test = 
train_test_split(X,y,test_size = 1/3, random_state = 0)

#fitting simple linear regression to the training set
from sklearn.linear_model import
LinearRegression
regressor = LinearRegression()
regressor.fix(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#visualising the training set results
plt.scatter(X_train, y_train, color='red)
plt.plot(X_train, regressor.predict(X_train), color='blue)
plt.tittle('Salary Vs Exper)