# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 09:35:31 2019

@author: 30spartan0
"""

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from scipy.stats import randint
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


#read in the excel file
data = pd.read_excel('Real estate valuation data set.xlsx', index_col='No')

#sanity check
print(data.head())

#explore the data set
print(data.info())  #we find all values are non-null
print(data.shape)   # 414 rows & 7 columns
print(data.describe()) 

#Simple EDA 
sns.lmplot(x='X2 house age', y='Y house price of unit area', data=data) 
plt.show()

sns.lmplot(x='X3 distance to the nearest MRT station', y='Y house price of unit area', data=data)
plt.show()

sns.lmplot(x='X4 number of convenience stores', y='Y house price of unit area', data=data)
plt.show()

maxpos = data['Y house price of unit area'].idxmax()
print('max position of y-value is: ' + str(maxpos))
data_wo_outlier = data.drop(data.index[270]) #we drop the outlier data point(index starts with 1 so have to minus 1)

sns.lmplot(x='X2 house age', y='Y house price of unit area', data=data_wo_outlier) 
plt.show()

"""
Above plots support the simple hypotheses that younger houses, houses nearer to MRT & with higher no.of
convenience stores are generally more expensive. Also outlier with exceptionally high price is spotted. 
"""
#we can start to fit our data into models to train and thus determine certain metrics for R2 and RMSE
y = data_wo_outlier['Y house price of unit area']
X = data_wo_outlier.drop(['Y house price of unit area'], axis=1)

#sanity
print(y.head())
print(y.shape)
print(X.info())
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

#we fit with a simple linear regressor to find out the the initial R2 score and RMSE. Main metric will be R2 score
reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

#print the scores
print("Initial R2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print("Root Mean Squared Error: {}".format(rmse))
print("Mean Absolute Error: {}".format(mae))
"""
We now have our intial R2 and RMSE scores for our linear regression model. However, we still need to
cross-validate our model & finetune the model to improve the scores. We do this through CV from sklearn and also by fitting 
different models

"""
#doing a 5-fold CV with linear regression helps to improve the R2 score
cv_results = cross_val_score(reg, X, y, cv=5)
print(cv_results)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_results)))

#Lasso visualization helps to show that convenience store location & transaction date play the most impt part for feature selection
names = X.columns
lasso = Lasso(alpha=0.1)

lasso_coef = lasso.fit(X, y).coef_
_= plt.plot(range(len(names)), lasso_coef)
_= plt.xticks(range(len(names)), names, rotation=60)
_= plt.ylabel('Co-efficients')
plt.show()

#Ridge regression to see if there is improvement in R2 score 
ridge = Ridge(alpha = 0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print(ridge.score(X_test, y_test))
ridge_cv = cross_val_score(ridge, X, y, cv=5)

print("Mean Ridge score: {}".format(np.mean(ridge_cv)))

#We move to decsion tree and random forest and tune the parameters for best scores 

#Parameters that we wish to tune for best score
param_dist = {"max_depth": [3, None],
              "min_samples_leaf": randint(1, 9),
              "criterion": ["mse"]}

#Tree classifier
tree = DecisionTreeRegressor()
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
tree_cv.fit(X_train,y_train)

#Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best Decision Tree score is {}".format(tree_cv.best_score_))


#We now go for random forest as decision tree regressor has improved the R2 score

#Forest Classifier
forest = RandomForestRegressor()

#Parameters that we wish to tune 
param_dist2 = {"n_estimators" : np.arange(1,100),
               "criterion" : ["mse", "mae"],
               "min_samples_leaf": randint(1, 9)
        }
forest_cv = RandomizedSearchCV(forest, param_dist2, cv=5)
forest_cv.fit(X_train,y_train)

#Print the tuned parameters and best scores. We notice an improvement in the score.
print("Tuned Forest Parameters: {}".format(forest_cv.best_params_))
print("Best Forest score is {}".format(forest_cv.best_score_))

#Finally we add a small pipeline process with a scaling process to further improve the score
pl = Pipeline([('scale', MaxAbsScaler()), ('forest', RandomForestRegressor())])
pl.fit(X_train, y_train)

y_pl = pl.predict(X_test)
rmse_forest = np.sqrt(mean_squared_error(y_test, y_pl))
mae_forest = mean_absolute_error(y_test, y_pl)
print("Final R2 Score: {}".format(pl.score(X_test, y_test)))
print("Final RMSE score: {}".format(rmse_forest))
print("Final MAE score: {}".format(mae_forest))