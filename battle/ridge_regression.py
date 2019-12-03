# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import Ridge, SGDRegressor

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
x_test = np.array(1.5).reshape(-1, 1)

ridge_reg = Ridge(alpha=1.0, solver='sag')
ridge_reg.fit(X, y)

print(ridge_reg.intercept_)
print(ridge_reg.coef_)

theta_ridge = np.array([ridge_reg.intercept_, ridge_reg.coef_])

y_test = ridge_reg.predict(x_test)

print(y_test)

sgd_reg = SGDRegressor(penalty='l2', n_iter_no_change=900)
sgd_reg.fit(X, y.ravel())

print('w0 -> {}'.format(sgd_reg.intercept_))
print('w1 -> {}'.format(sgd_reg.coef_))

theta_sgd_l2 = np.array([sgd_reg.intercept_, sgd_reg.coef_])

y_test = ridge_reg.predict(x_test)

print(y_test)
