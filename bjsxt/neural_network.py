# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([[0, 0], [1, 1]])
y = np.array([0, 1]).reshape(-1, 1)

clf = MLPClassifier(hidden_layer_sizes=(5, 2), activation='logistic', solver='sgd',
                    alpha=1e-5, max_iter=2000, tol=1e-4)
clf.fit(X, y)

coef = clf.coefs_

X_test = np.array([[2, 2], [-1, -2]])

y_predict = clf.predict(X_test)
y_pred_proba = clf.predict(X_test)
