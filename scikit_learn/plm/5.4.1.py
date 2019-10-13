# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import numpy as np

from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

boston = load_boston()
X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
X_test = X_scaler.transform(X_test)
y_test = y_scaler.transform(y_test.reshape(-1, 1))

regressor = SGDRegressor()
scores = cross_val_score(regressor, X_train, y_train, cv=5)
scores_mean = scores.mean()

print('Cross validation r-squared scores: {}'.format(scores))
print('Average cross validation r-squared score: {}'.format(scores_mean))

regressor.fit(X_train, y_train)
test_score = regressor.score(X_test, y_test)

print('Test set r-squared score: {}'.format(test_score))
