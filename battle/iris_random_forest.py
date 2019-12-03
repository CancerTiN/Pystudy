# -*- coding: utf-8 -*-

import logging

from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
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

logger.info('Training data with parameters as follow:\n{}'.format(
    {'test_size': 0.33, 'random_state': 42, 'n_estimators': 15, 'max_leaf_nodes': 16}))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rnd_clf = RandomForestClassifier(n_estimators=15, max_leaf_nodes=16)
rnd_clf.fit(X_train, y_train)

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter='random', max_leaf_nodes=16),
    n_estimators=15, max_samples=1.0
)
bag_clf.fit(X_train, y_train)

y_pred_rnd = rnd_clf.predict(X_test)
y_pred_bag = bag_clf.predict(X_test)

acc_rnd = accuracy_score(y_test, y_pred_rnd)
acc_bag = accuracy_score(y_test, y_pred_bag)
pre_rnd = precision_score(y_test, y_pred_rnd, average='macro')
pre_bag = precision_score(y_test, y_pred_bag, average='macro')
rec_rnd = recall_score(y_test, y_pred_rnd, average='macro')
rec_bag = recall_score(y_test, y_pred_bag, average='macro')
f1s_rnd = f1_score(y_test, y_pred_rnd, average='macro')
f1s_bag = f1_score(y_test, y_pred_bag, average='macro')

logger.info('Rnd Accuracy score: {}'.format(acc_rnd))
logger.info('Bag Accuracy score: {}'.format(acc_bag))
logger.info('Rnd Precision score: {}'.format(pre_rnd))
logger.info('Bag Precision score: {}'.format(pre_bag))
logger.info('Rnd Recall score: {}'.format(rec_rnd))
logger.info('Bag Recall score: {}'.format(rec_bag))
logger.info('Rnd F1 score: {}'.format(f1s_rnd))
logger.info('Bag F1 score: {}'.format(f1s_bag))

rnd_clf = RandomForestClassifier(n_estimators=500)
rnd_clf.fit(data, target)
for name, importance in zip(iris.feature_names, rnd_clf.feature_importances_):
    logger.info('Get {} importance from feature ({})'.format(importance, name))
