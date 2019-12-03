# -*- coding: utf-8 -*-

import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

learning_rate = 0.1
n_iterations = 100000
m = 100

theta = np.random.randn(2, 1)
count = 0

for iteration in range(n_iterations):
    count += 1
    gradients = 1 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients
