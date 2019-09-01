# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

from sklearn import preprocessing
import numpy as np

X = np.array([
    [0, 0,  5, 13,  9,  1],
    [0, 0, 13, 15, 10, 15],
    [0, 3, 15,  2,  0, 11],
])
X_scale = preprocessing.scale(X)

print('X:\n{}'.format(X))
print('X scale:\n{}'.format(X_scale))
