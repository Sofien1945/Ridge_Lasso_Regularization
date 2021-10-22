"""Ridge and lasso regularization of linear regression
Part of Sipleran Machine Learning Course
Date: 22.10.2021
Done By Sofien ABidi"""

#Import standard + Dataset + Linear regression libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#%matplotlib inline

#Dataset attribution
boston_dataset = load_boston()
print(boston_dataset.keys())
boston_dataset.DESCR
boston_pd = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston_pd['House Price'] = boston_dataset.target

#Input
X = boston_pd.iloc[:,:-1]

#Output
y = boston_pd.iloc[:,-1]

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

#Fit regeresion model
lreg = LinearRegression()
lreg.fit(X_train, y_train)

#Make Prediction using test data
y_pred = lreg.predict(X_test)

#Calculationg mean squared errors MSE
mean_squarred_error = np.mean((y_pred - y_test)**2)
print('Mean squared error= ', mean_squarred_error)

#Coefficient Calculation
lreg_coef = pd.DataFrame()
lreg_coef['Columns'] = X_train.columns
lreg_coef['Coefficient Estimate'] = pd.Series(lreg.coef_)
print(lreg_coef)

#print Coefficient score
fig, ax = plt.subplots(figsize=(20,10))
ax.bar(lreg_coef['Columns'], lreg_coef['Coefficient Estimate'],color=['purple', 'red', 'green', 'blue', 'cyan','yellow', 'green','brown'],edgecolor='black')
ax.spines['bottom'].set_position('zero')
plt.style.use('ggplot')

#Ridge regression to minimaze coefficent variance
from sklearn.linear_model import Ridge

#Train the model
rdg = Ridge(alpha=1)
rdg.fit(X_train, y_train)
y_pred_r = rdg.predict(X_test)

#Calculationg mean squared errors MSE of ridge
mean_squarred_error = np.mean((y_pred_r - y_test)**2)
print('Mean squared error= ', mean_squarred_error)

#Coefficient Calculation of ridge
lreg_coef_r = pd.DataFrame()
lreg_coef_r['Columns'] = X_train.columns
lreg_coef_r['Coefficient Estimate'] = pd.Series(rdg.coef_)
print(lreg_coef_r)

#Lasso regression to minimaze coefficent variance
from sklearn.linear_model import Lasso

#Train the model
las = Lasso(alpha=1)
las.fit(X_train, y_train)
y_pred_l = las.predict(X_test)

#Calculationg mean squared errors MSE of Lasso
mean_squarred_error = np.mean((y_pred_l - y_test)**2)
print('Mean squared error= ', mean_squarred_error)

#Coefficient Calculation of Lasso
lreg_coef_l = pd.DataFrame()
lreg_coef_l['Columns'] = X_train.columns
lreg_coef_l['Coefficient Estimate'] = pd.Series(las.coef_)
print(lreg_coef_l)

#print Coefficient score
fig, ax = plt.subplots(figsize=(20,10))
ax.bar(lreg_coef_l['Columns'], lreg_coef_l['Coefficient Estimate'],color=['purple', 'red', 'green', 'blue', 'cyan','yellow', 'green','brown'],edgecolor='black')
ax.spines['bottom'].set_position('zero')
plt.style.use('ggplot')
plt.show()

