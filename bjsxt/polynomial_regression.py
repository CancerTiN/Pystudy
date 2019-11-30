# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + 2 * X + 2 + np.random.randn(m, 1)

plt.plot(X, y, 'b.')
for degree, fmt in {1: 'g-', 2: 'r+', 10: 'y*'}.items():
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    y_predict = lin_reg.predict(X_poly)
    plt.plot(X_poly[:, 0], y_predict, fmt)
plt.show()
