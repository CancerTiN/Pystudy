# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X_train = np.array([6, 8, 10, 14,   18]).reshape(-1, 1)
y_train = np.array([7, 9, 13, 17.5, 18]).reshape(-1, 1)
X_test = np.array([6, 8,  11, 16]).reshape(-1, 1)
y_test = np.array([8, 12, 15, 18]).reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100).reshape(-1, 1)
yy = regressor.predict(xx)
plt.plot(xx, yy)

quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
X_test_quadratic = quadratic_featurizer.transform(X_test)

regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx)
yy_quadratic = regressor_quadratic.predict(xx_quadratic)
plt.plot(xx, yy_quadratic, c='r', linestyle='--')
plt.title('Pizza price regressed on diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(X_train, y_train)
plt.show()

print('X_train: {}'.format(X_train))
print('X_train_quadratic: {}'.format(X_train_quadratic))
print('X_test: {}'.format(X_test))
print('X_test_quadratic: {}'.format(X_test_quadratic))

print('Simple linear regression r-squared: {}'.format(
    regressor.score(X_test, y_test)
))
print('Quadratic regression r-squared: {}'.format(
    regressor_quadratic.score(X_test_quadratic, y_test)
))
