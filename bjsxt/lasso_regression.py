# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import Lasso, SGDRegressor

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
x_test = np.array(1.5).reshape(-1, 1)

lasso_reg = Lasso(alpha=0.15)
lasso_reg.fit(X, y)

y_test = lasso_reg.predict(x_test)

print(lasso_reg.intercept_)
print(lasso_reg.coef_)

theta_lasso = np.array([lasso_reg.intercept_, lasso_reg.coef_])

print(y_test)

sgd_reg = SGDRegressor(penalty='l1', n_iter_no_change=900)
sgd_reg.fit(X, y.ravel())

print('w0 -> {}'.format(sgd_reg.intercept_))
print('w1 -> {}'.format(sgd_reg.coef_))

theta_sgd_l1 = np.array([lasso_reg.intercept_, lasso_reg.coef_])

y_test = lasso_reg.predict(x_test)

print(y_test)
