# -*- coding: utf-8 -*-

import numpy as np

from sklearn.svm import SVC

X = [[10, 21], [20, 11], [13, 10], [10, 10], [20, 10]]
y = [1, 1, -1, -1, 1]
model = SVC(kernel='linear', gamma='scale', probability=True)
model.fit(X, y)
label1, proba1 = model.predict([[25, 25]]), model.predict_proba([[25, 25]])
label2, proba2 = model.predict([[25, 20]]), model.predict_proba([[25, 20]])
print('Label1: {}; Label2: {}'.format(label1, label2))
print('Proba1: {}; Proba2: {}'.format(proba1, proba2))

print(np.dot(model.coef_, np.array([10, 21]).reshape(2, 1)) + model.intercept_)
print(np.dot(model.coef_, np.array([20, 11]).reshape(2, 1)) + model.intercept_)
print(np.dot(model.coef_, np.array([13, 10]).reshape(2, 1)) + model.intercept_)
print(np.dot(model.coef_, np.array([10, 10]).reshape(2, 1)) + model.intercept_)
print(np.dot(model.coef_, np.array([20, 10]).reshape(2, 1)) + model.intercept_)

import matplotlib.pyplot as plt

xc = [i[0] for i in X]
yc = [i[1] for i in X]
xlc = np.linspace(0, 30, 1000)
ylc = (-model.intercept_[0] - model.coef_[0][0] * xlc) / model.coef_[0][1]
plt.figure()
plt.scatter(xc, yc, c=y)
for support_vector in model.support_vectors_:
    circle = plt.Circle(support_vector, 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
    plt.gca().add_patch(circle)
plt.plot(xlc, ylc, 'r-')
plt.xlim((5, 25))
plt.ylim((5, 25))
plt.show()
