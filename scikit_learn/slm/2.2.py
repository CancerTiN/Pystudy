# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import numpy as np
from sklearn.linear_model import LinearRegression

X_train = np.array([6, 8, 10, 14, 18]).reshape(-1, 1)
print('explanatory variable: {}'.format(X_train))
y_train = [7, 9, 13, 17.5, 18]
print('response variable: {}'.format(y_train))

X_test = np.array([8, 9, 11, 16, 12]).reshape(-1, 1)
y_test = [11, 8.5, 15, 18, 11]

model = LinearRegression()
model.fit(X_train, y_train)
r_squared = model.score(X_test, y_test)
print('R squared: {}'.format(r_squared))
