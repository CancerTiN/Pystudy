# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import numpy as np
from scipy.spatial.distance import euclidean

X_train = np.array([
    [1700, 1],
    [1600, 0]
])
x_test = np.array([1640, 1])
print('distance between {} and {} is {}'.format(X_train[0,:], x_test, euclidean(X_train[0,:], x_test)))
print('distance between {} and {} is {}'.format(X_train[1,:], x_test, euclidean(X_train[1,:], x_test)))

X_train = np.array([
    [1.7, 1],
    [1.6, 0]
])
x_test = np.array([1.64, 1])
print('distance between {} and {} is {}'.format(X_train[0,:], x_test, euclidean(X_train[0,:], x_test)))
print('distance between {} and {} is {}'.format(X_train[1,:], x_test, euclidean(X_train[1,:], x_test)))
