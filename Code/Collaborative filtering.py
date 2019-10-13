# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Python Code:
#Code for svd

import os
import pandas as pd
import surprise
import numpy as np
import sklearn
import matplotlib.pyplot as plt   
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from surprise.model_selection import cross_validate


train = pd.read_csv("train_data.csv")
train_label = pd.read_csv("train_labels.csv")
test = pd.read_csv("test_data.csv")
test_label = pd.read_csv("test_labels.csv")
print("Training Set Loaded")
algo = surprise.SVD() # Singular Vector Decomposition Function
reader = surprise.Reader(rating_scale=(0, 1)) # To parse the file and ratings as 0 or 1 since the labels are binary
dat = train[['msno', 'song_id']]
dat = pd.concat([dat, train_label], axis=1)
dat2 = test[['msno', 'song_id']]
dat2 = pd.concat([dat2, test_label], axis=1)
data = surprise.Dataset.load_from_df(dat.dropna(), reader)
data2 = surprise.Dataset.load_from_df(dat2.dropna(), reader)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)
trainset = data.build_full_trainset()
testset = data2.build_full_trainset()
msno = test[['msno']]
song_id = test[['song_id']]

msno = str(44)
song_id = str(44)

algo.fit(trainset)
print(algo.predict(msno,song_id))

testset2 = testset.build_testset()
predictions = algo.test(testset2)
# RMSE should be low as we are biased
# RMSE should be low as we are biased
surprise.accuracy.mae(predictions, verbose=True)
surprise.accuracy.rmse(predictions, verbose=True)


submit = []
from statistics import mean
for index, row in test.iterrows():
	est = algo.predict(row['msno'], row['song_id']).est
	submit.append(int(round(est)))

print(mean(test_label['x'] != submit)) # misclassification rate (test error)

