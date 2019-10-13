# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 00:51:08 2019

@author: User
"""
CODE FOR LOG REGRESSION
import os
from sklearn import cross_validation, grid_search, metrics, ensemble
import sklearn.linear_model as lr
from statistics import mean
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


warnings.filterwarnings('ignore')


df = pd.read_csv("df.csv")
test_l = pd.read_csv("test_l.csv")
test_d = pd.read_csv("test_d.csv")
target = df['target']
df = df.drop(['target'],1)
# Object data to category
for col in df.select_dtypes(include=['object']).columns:
	df[col] = df[col].astype('category')
    
# Encoding categorical features
for col in df.select_dtypes(include=['category']).columns:
	df[col] = df[col].cat.codes

# Object data to category
for col in test_d.select_dtypes(include=['object']).columns:
	test_d[col] = test_d[col].astype('category')
    
# Encoding categorical features
for col in test_d.select_dtypes(include=['category']).columns:
	test_d[col] = test_d[col].cat.codes

train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(df, target, test_size = 0.3, random_state=1)
print('Done splitting!')

# Basic Logistic Regression
print('\n','Logistic Regression')
model = lr.LogisticRegression()
model.fit(train_data, train_labels) # training model
# Predicting
predict_labels = model.predict(test_data) # predicting model with pseudo test data via CV
print(metrics.classification_report(test_labels, predict_labels))
mispred_counts = (predict_labels != test_labels).sum()
test_error = mispred_counts / len(predict_labels) # misclassification error (pseudo test error)
print('Pseudo Test error = ', test_error)

# actual test error
actual_test_labels = test_l['x'] # removing labels on test data
actual_test_data = test_d
predict_labels = model.predict(actual_test_data) # predicting model with test data
print(metrics.classification_report(actual_test_labels, predict_labels))
actual_test_error = mean(predict_labels != actual_test_labels)
print('Actual Test error = ', actual_test_error)


print("(Pseudo Test) MSE = ",mean_squared_error(test_labels, model.predict(test_data)))
print("(Pseudo Test) MAE = ",mean_absolute_error(test_labels, model.predict(test_data)))
print("(Test data) MSE = ",mean_squared_error(actual_test_labels, model.predict(actual_test_data)))
print("(Test data) MAE = ",mean_absolute_error(actual_test_labels, model.predict(actual_test_data)))

print("(Pseudo Test data) AUC ROC Score = ",roc_auc_score(model.predict(test_data),test_labels))
print("(Test data) AUC ROC Score = ",roc_auc_score(predict_labels,actual_test_labels))
x = model.fit(train_data, train_labels)
# print('model.classes_ =', x.classes_)
# coefficients of basic regression model
print('model.coef_ =', x.coef_)

max_features = 100
# corresponding features in basic regression model
train_data.iloc[0]

# Logistics Regression W Lasso (Lambda obtained via CV)
# 10-fold CV to choose the best alpha
# refit the model and compute the test error and model coefficients
alphas = 10**np.linspace(6,-10,50)*0.5 # Lambdas=alphas
lasso = Lasso(max_iter=10000, normalize=True)
coefs = []
for a in alphas:
	lasso.set_params(alpha=a) #set L1 penalty lambda parameter = a
	lasso.fit(train_data, train_labels)
	coefs.append(lasso.coef_) #obtain coefficients for log reg model at every lambda
    
lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)
lassocv.fit(train_data, train_labels)
lasso = Lasso(max_iter=10000, normalize=True)
lasso.set_params(alpha=lassocv.alpha_) #obtain the best lambda
print("Lambda=", lassocv.alpha_)
lasso.fit(train_data, train_labels)
print("(Pseudo Test) MSE = ",mean_squared_error(test_labels, lasso.predict(test_data)))
print("(Pseudo Test) MAE = ",mean_absolute_error(test_labels, lasso.predict(test_data)))
print("(Test) MSE = ",mean_squared_error(actual_test_labels, lasso.predict(actual_test_data)))
print("(Test) MAE = ",mean_absolute_error(actual_test_labels, lasso.predict(actual_test_data)))
print("best model coefficients:")
pd.Series(lasso.coef_, index=train_data.columns)

# Cross validation Ridge
ridgecv = RidgeCV(alphas=alphas, normalize=True)
ridgecv.fit(train_data, train_labels)
print("Alpha=", ridgecv.alpha_)
ridge6 = Ridge(alpha=ridgecv.alpha_, normalize=True)
ridge6.fit(train_data, train_labels)
print("psuedo mse = ",mean_squared_error(test_labels, ridge6.predict(test_data)))
print("mse = ",mean_squared_error(actual_test_labels, ridge6.predict(actual_test_data)))
print("mae = ",mean_absolute_error(actual_test_labels, ridge6.predict(actual_test_data)))
print("best model coefficients:")
pd.Series(ridge6.coef_, index=train_data.columns)

# actual test error for lasso
actual_test_data = test_d
predict_labels = lasso.predict(actual_test_data)
Result = []
for i in range(len(predict_labels)):
	if predict_labels[i] >= 0.5:
    	Result.append(1)
	else:
    	Result.append(0)
print(metrics.classification_report(actual_test_labels, Result))
actual_test_error = mean(Result != actual_test_labels)
print('Actual Test error = ', actual_test_error)

# actual test error for ridge
actual_test_data = test_d
predict_labels = ridge6.predict(actual_test_data)
Result = []
for i in range(len(predict_labels)):
	if predict_labels[i] >= 0.5:
    	Result.append(1)
	else:
    	Result.append(0)
print(metrics.classification_report(actual_test_labels, Result))
actual_test_error = mean(Result != actual_test_labels)
print('Actual Test error = ', actual_test_error)

