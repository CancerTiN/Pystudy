# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import Ridge, SGDRegressor

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
x_test = np.array(1.5).reshape(-1, 1)

rideg_reg = Ridge(alpha=1.0, solver='sag')
rideg_reg.fit(X, y)

print(rideg_reg.intercept_)
print(rideg_reg.coef_)

y_test = rideg_reg.predict(x_test)

print(y_test)

sgd_reg = SGDRegressor(penalty='l2', n_iter_no_change=900)
sgd_reg.fit(X, y.ravel())

print('w0 -> {}'.format(sgd_reg.intercept_))
print('w1 -> {}'.format(sgd_reg.coef_))

y_test = rideg_reg.predict(x_test)

print(y_test)
