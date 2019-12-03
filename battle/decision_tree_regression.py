# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

N = 100
X = np.random.rand(N) * 6 - 3
X.sort()
y = np.sin(X) + np.random.randn(N) * 0.05
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

dt_reg = DecisionTreeRegressor()
dt_reg.fit(X.reshape(-1, 1), y.reshape(-1, 1))

X_test = np.linspace(np.around(min(X)), np.ceil(max(X)), 100).reshape(-1, 1)
y_pred = dt_reg.predict(X_test)

plt.plot(X, y, 'y*', label='actual')
plt.plot(X_test, y_pred, 'b-', label='predict')
plt.grid()
plt.show()
plt.close()

plt.plot(X, y, 'ko', label='actual')
for depth, color in zip(range(2, 11, 2), 'rgbmy'):
    dt_reg.set_params(max_depth=depth)
    dt_reg.fit(X, y)
    y_pred = dt_reg.predict(X_test)
    plt.plot(X_test, y_pred, '-', color=color, label='depth={}'.format(depth))
plt.legend()
plt.grid(b=True)
plt.show()
