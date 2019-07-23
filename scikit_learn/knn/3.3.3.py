# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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
print('response variable: {}'.format(y_train))

lb = LabelBinarizer()
y_train_binarized = lb.fit_transform(y_train)
print('response variable binarized: {}'.format(y_train_binarized))
K = 3
print('n neighbors: {}'.format(K))
clf = KNeighborsClassifier(n_neighbors=K)
clf.fit(X_train, y_train_binarized.reshape(-1))

X_test = np.array([
    [168, 65],
    [180, 96],
    [160, 52],
    [169, 67],
])
print('X test variable: {}'.format(X_test))
y_test = ['male', 'male', 'female', 'female']
print('y test variable: {}'.format(y_test))
y_test_binarized = lb.transform(y_test)
print('binarized labels: {}'.format(y_test_binarized))
predictions_binarized = clf.predict(X_test)
print('binarized predictions: {}'.format(predictions_binarized))
predictions_labels = lb.inverse_transform(predictions_binarized)
print('predicted labels: {}'.format(predictions_labels))

Accuracy = accuracy_score(y_test_binarized, predictions_binarized)
print('accuracy: {}'.format(Accuracy))
