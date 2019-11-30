# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import ElasticNet, SGDRegressor

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
x_test = np.array(1.5).reshape(-1, 1)

elastic_net = ElasticNet(alpha=0.0001, l1_ratio=0.15)
elastic_net.fit(X, y)

y_test = elastic_net.predict(x_test)

print(elastic_net.intercept_)
print(elastic_net.coef_)

print(y_test)

sgd_reg = SGDRegressor(penalty='elasticnet', n_iter_no_change=900)
sgd_reg.fit(X, y.ravel())

print('w0 -> {}'.format(sgd_reg.intercept_))
print('w1 -> {}'.format(sgd_reg.coef_))

y_test = elastic_net.predict(x_test)

print(y_test)
