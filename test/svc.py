# -*- coding: utf-8 -*-

from sklearn.svm import SVC

X = [[10, 21], [20, 11], [13, 10], [10, 10], [20, 10]]
y = [1, 1, -1, -1, 1]
model = SVC(kernel='linear', gamma='scale')
model.fit(X, y)

print(np.dot(model.coef_, np.array([10, 21]).reshape(2, 1)) + model.intercept_)
print(np.dot(model.coef_, np.array([20, 11]).reshape(2, 1)) + model.intercept_)
print(np.dot(model.coef_, np.array([13, 10]).reshape(2, 1)) + model.intercept_)
print(np.dot(model.coef_, np.array([10, 10]).reshape(2, 1)) + model.intercept_)
print(np.dot(model.coef_, np.array([20, 10]).reshape(2, 1)) + model.intercept_)

import matplotlib.pyplot as plt

xc = [i[0] for i in X]
yc = [i[1] for i in X]
plt.figure()
plt.scatter(xc, yc, c=y)
plt.show()
