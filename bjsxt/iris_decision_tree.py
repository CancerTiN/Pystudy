# -*- coding: utf-8 -*-

import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

iris = load_iris()

data = iris['data']
target = iris['target']

X = data[:, :2]
y = target

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=8)
tree_clf.fit(x_train, y_train)

y_hat = tree_clf.predict(x_test)

acc = accuracy_score(y_test, y_hat)
logger.info('Accuracy score: {}'.format(acc))

x_self = np.array([[5, 1.5]])
y_self = tree_clf.predict(x_self)
y_self_proba = tree_clf.predict_proba(x_self)

errs = list()
for depth in range(1, 15):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    clf.fit(x_train, y_train)
    y_hat = clf.predict(x_test)
    score = (y_test == y_hat)
    err = 1 - np.mean(score)
    errs.append(err)
    logger.info('Get {} error rate with max depth = {}'.format(err, depth))

plt.plot(range(1, 15), errs, 'ro-', lw=2)
plt.xlabel('Max depth')
plt.ylabel('Error rate')
plt.grid(True)
plt.waitforbuttonpress()
plt.close()
