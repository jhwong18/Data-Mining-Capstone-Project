# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 00:51:23 2019

@author: User
"""

CODE FOR TREES:

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
import lightgbm as lgbm

warnings.filterwarnings('ignore')


train_labels = pd.read_csv("train_l.csv")
train_data = pd.read_csv("train_d.csv")
train_data = train_data.drop(['target'],1)
test_labels = pd.read_csv("test_l.csv")
test_data = pd.read_csv("test_d.csv")

# Object data to category
for col in train_data.select_dtypes(include=['object']).columns:
	train_data[col] = train_data[col].astype('category')
    
# Encoding categorical features
for col in train_data.select_dtypes(include=['category']).columns:
	train_data[col] = train_data[col].cat.codes

# Object data to category
for col in test_data.select_dtypes(include=['object']).columns:
	test_data[col] = test_data[col].astype('category')
    
# Encoding categorical features
for col in test_data.select_dtypes(include=['category']).columns:
	test_data[col] = test_data[col].cat.codes


d_train = lgbm.Dataset(train_data,train_labels['x'])
watchlist = [d_train]

num_train, num_feature = train_data.shape
feature_name = ['feature_' + str(col) for col in range(num_feature)]

params = {}
params['learning_rate'] = 0.5
params['application'] = 'binary'
params['max_depth'] = 10
params['num_leaves'] = 2**6
params['verbosity'] = 0
params['metric'] = 'auc'

model = lgbm.train(params, train_set=d_train, num_boost_round=250, valid_sets=watchlist, verbose_eval=50,
               	feature_name = feature_name, categorical_feature=[0,1,2,4,5,6,7,8])

y_pred = model.predict(test_data, num_iteration=model.best_iteration)
y_pred = (y_pred > 0.5)
y_pred.astype(int)
pred_count = (y_pred == test_labels['x']).sum()
pred_perc = pred_count/len(y_pred)
#error is
print(1-pred_perc)

# Load libraries
from sklearn import cross_validation, grid_search, metrics, ensemble
import sklearn.linear_model as lr
from statistics import mean
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import warnings
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

# actual test error
actual_test_labels = test_l['x']
actual_test_data = test_d

# xgboost
print('\n','XGBoost')
from sklearn.grid_search import GridSearchCV
optimization_dict = {
                	'max_depth': [2,4,6,10],
                 	'n_estimators': [50,100,200,250]}
model = xgb.XGBClassifier()
model = GridSearchCV(model, optimization_dict,
                 	scoring='accuracy', verbose=1)

model.fit(train_data, train_labels)
print(model.best_score_)
print(model.best_params_)

#model = xgb.XGBClassifier(learning_rate=0.1, max_depth=6, min_child_weight=5, n_estimators=50)
#model.fit(train_data, train_labels)


# Predicting
predict_labels = model.predict(test_data)
print(metrics.classification_report(test_labels, predict_labels))
test_error = mean(predict_labels != test_labels)
print('Test error = ', test_error)

# actual test error
predict_labels = model.predict(actual_test_data)
print(metrics.classification_report(actual_test_labels, predict_labels))
actual_test_error = mean(predict_labels != actual_test_labels)
print('Actual Test error = ', actual_test_error)

# random forest
from sklearn.grid_search import GridSearchCV
optimization_dict = {
                	'max_depth': [2,4,6,10],
                 	'n_estimators': [50,100,200,250]}
model = ensemble.RandomForestClassifier()
model = GridSearchCV(model, optimization_dict,
                 	scoring='accuracy', verbose=1)

model.fit(train_data, train_labels)
print(model.best_score_)
print(model.best_params_)
##model = ensemble.RandomForestClassifier(n_estimators=250, max_depth=15)
print('Fitting...')
##model.fit(train_data, train_labels)
# Predicting
print('Predicting...')
predict_labels = model.predict(test_data)
print('\n','Random Forest')
print(metrics.classification_report(test_labels, predict_labels))
test_error = mean(predict_labels != test_labels)
print('Test error = ', test_error)

# actual test error
predict_labels = model.predict(actual_test_data)
print(metrics.classification_report(actual_test_labels, predict_labels))
actual_test_error = mean(predict_labels != actual_test_labels)
print('Actual Test error = ', actual_test_error)

std = np.std([tree.feature_importances_ for tree in model.estimators_],
         	axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
X = train_data

for f in range(X.shape[1]):
	print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
   	color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
