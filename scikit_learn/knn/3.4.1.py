# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

X_train = np.array([
    [158, 1],
    [170, 1],
    [183, 1],
    [191, 1],
    [155, 0],
    [163, 0],
    [180, 0],
    [158, 0],
    [170, 0],
])
y_train = [64, 86, 84, 80, 49, 59, 67, 54, 67]

X_test = np.array([
    [168, 1],
    [180, 1],
    [160, 0],
    [169, 0],
])
y_test = np.array([65, 96, 52, 67])
print('Test weights: {}'.format(y_test))

K = 3
clf = KNeighborsRegressor(n_neighbors=K)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Predicted weights: {}'.format(y_pred))
print('Coefficient of determination: {}'.format(r2_score(y_test, y_pred)))
print('Mean absolute error: {}'.format(mean_absolute_error(y_test, y_pred)))
print('Mean squared error: {}'.format(mean_squared_error(y_test, y_pred)))
