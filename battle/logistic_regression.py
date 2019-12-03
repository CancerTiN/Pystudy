# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

X = iris['data'][:, 3].reshape(-1, 1)
y = iris['target']

log_reg = LogisticRegression(solver='sag', multi_class='multinomial')
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)

y_hat = log_reg.predict(X_new)
y_proba = log_reg.predict_proba(X_new)

plt.plot(X_new, y_proba[:, 0], 'b-', label='Iris-Setosa')
plt.plot(X_new, y_proba[:, 1], 'r-', label='Iris-Versicolour')
plt.plot(X_new, y_proba[:, 2], 'g-', label='Iris-Virginica')
plt.show()
