# -*- coding: utf-8 -*-

import logging

from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

iris = load_iris()

X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)])

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info('Model {} recevie {} accuracy'.format(clf.__class__.__name__, acc))

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, oob_score=True)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
y_pred_proba = bag_clf.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)

logger.info('Prediction probability: {}'.format(y_pred_proba))
logger.info('Accuracy score: {}'.format(acc))
logger.info('Oob score: {}'.format(bag_clf.oob_score_))
logger.info('Oob decision function: {}'.format(bag_clf.oob_decision_function_))
