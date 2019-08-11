# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

X_train = np.array([
    [158, 64],
    [170, 86],
    [183, 84],
    [191, 80],
    [155, 49],
    [163, 59],
    [180, 67],
    [158, 54],
    [170, 67],
])
print('explanatory variable: {}'.format(X_train))
y_train = ['male'] * 4 + ['female'] * 5
y_train = np.array(y_train)
print('response variable: {}'.format(y_train))

plt.figure()
plt.title('Human Heights and Weights by sex')
plt.xlabel('Height in cm')
plt.ylabel('Height in kg')
for i, x in enumerate(X_train):
    plt.scatter(x[0], x[1], c='k', marker='x' if y_train[i] == 'male' else 'D')
plt.grid(True)
plt.show()

x = np.array([[155, 70]])
distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
print('distances: {}'.format(distances))

nearest_neighbor_indices = distances.argsort()[:3]
print('nearest neighbor indices: {}'.format(nearest_neighbor_indices))
nearest_neighbor_genders = np.take(y_train, nearest_neighbor_indices)
print('nearest neighbor genders: {}'.format(nearest_neighbor_genders))

b = Counter(nearest_neighbor_genders)
print('predict gender: {}'.format(b.most_common(1)))
print('predict gender: {}'.format(b.most_common(1)[0][0]))
