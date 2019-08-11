# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

[[6.9, 10, 10.5],
 [7.8, 12, 11.6],
 [8.3, 14, 12.9],
 [9.1, 16, 14.3],
 [9.5, 18, 17.5]]
[7, 9, 13, 17.5, 18]

X = np.array([[6.9, 10, 10.5],
              [7.8, 12, 11.6],
              [8.3, 14, 12.9],
              [9.1, 16, 14.3],
              [9.5, 18, 17.5]])
print('explanatory variable: {}'.format(X))
y = np.array([7, 9, 13, 17.5, 18])
print('response variable: {}'.format(y))

model = LinearRegression()
model.fit(X, y)
print('model: {}'.format(model))

xt = np.array([[10, 20, 20]])
yt = model.predict(xt)
print('yt: {}'.format(yt))
