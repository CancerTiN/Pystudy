# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
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
y_train = np.array(['male'] * 4 + ['female'] * 5)
print('response variable: {}'.format(y_train))

lb = LabelBinarizer()
y_train_binarized = lb.fit_transform(y_train)
print('response variable binarized: {}'.format(y_train_binarized))
print('response variable binarized: {}'.format(y_train_binarized.reshape(-1)))
K = 3
print('n neighbors: {}'.format(K))
clf = KNeighborsClassifier(n_neighbors=K)
clf.fit(X_train, y_train_binarized.reshape(-1))
predicted_binarized = clf.predict(np.array([[155, 70], [200, 100]]))
print('predicted binarized: {}'.format(predicted_binarized))
predicted_label = lb.inverse_transform(predicted_binarized)
print('predicted label: {}'.format(predicted_label))
