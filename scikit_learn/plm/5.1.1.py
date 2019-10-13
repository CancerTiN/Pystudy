# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import numpy as np

X = np.array([[1, 6, 2],
              [1, 8, 1],
              [1, 10, 0],
              [1, 14, 2],
              [1, 18, 0]])
y = np.array([7, 9, 13, 17.5, 18]).reshape(-1, 1)

print('X: {}'.format(X))
print('y: {}'.format(y))

beta = np.dot(
    np.linalg.inv(np.dot(np.transpose(X), X)),
    np.dot(np.transpose(X), y)
)

print('beta: {}'.format(beta))

ret = np.linalg.lstsq(X, y)
print('lstsq return: {}'.format(ret))
