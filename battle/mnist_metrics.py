# -*- coding: utf-8 -*-

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

mnist = fetch_mldata('MNIST original')

X = mnist['data']
y = mnist['target']

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
plt.axis('off')
plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[:60000]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

logger.info(y_test_5)
logger.info(y_test_5.shape)
logger.info(y_test_5.sum())

sgd_clf = SGDClassifier(loss='log', tol=1e-4, random_state=42)
sgd_clf.fit(X_train, y_train_5)

y_some_digit = sgd_clf.predict([some_digit])
logger.info(y_some_digit)

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_folds = X_train[test_index]
    y_test_folds = y_train_5[test_index]
    clone_clf = clone(sgd_clf)
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred_folds = clone_clf.predict(X_test_folds)
    acc = accuracy_score(y_test_folds, y_pred_folds)
    logger.info('Accuracy score: {}'.format(acc))

sgd_cvs = cross_val_score(sgd_clf, X_train, y_train_5, scoring='accuracy', cv=3)
logger.info('Cross validation score: {}'.format(sgd_cvs))


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
never_5_cvs = cross_val_score(never_5_clf, X_train, y_train_5, scoring='accuracy', cv=3)
logger.info('Cross validation score: {}'.format(never_5_cvs))

y_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
CM = confusion_matrix(y_train_5, y_pred)
logger.info('True positives: {}'.format(CM[0, 0]))
logger.info('False positives: {}'.format(CM[0, 1]))
logger.info('False negatives: {}'.format(CM[1, 0]))
logger.info('True negatives: {}'.format(CM[1, 1]))

tp = fp = 0
for i, j in zip(y_train_5, y_pred):
    if j:
        if i:
            tp += 1
        else:
            fp += 1
ppv_self = tp / (tp + fp)
ppv = precision_score(y_train_5, y_pred)
assert ppv_self == ppv
logger.info('Precision score: {}'.format(ppv))

tp = fn = 0
for i, j in zip(y_train_5, y_pred):
    if i:
        if j:
            tp += 1
        else:
            fn += 1
tpr_self = tp / (tp + fn)
tpr = recall_score(y_train_5, y_pred)
assert tpr_self == tpr
logger.info('Recall score: {}'.format(tpr))

f1_self = 2 * (ppv * tpr) / (ppv + tpr)
f1 = f1_score(y_train_5, y_pred)
logger.info('F1 score: {}'.format(f1))

sgd_clf.fit(X_train, y_train_5)
some_digit_proba = sgd_clf.decision_function([some_digit])[0]

y_score = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_score)

plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
plt.plot(thresholds, recalls[:-1], 'r--', label='recall')
plt.xlabel('threshold')
plt.ylim([0, 1])
plt.legend()
plt.show()

ppv_90_proba = np.where(precisions >= 0.9)[0][0]
y_ppv_90 = y_score > ppv_90_proba
ppv = precision_score(y_train_5, y_ppv_90)
logger.info('Precision score: {}'.format(ppv))
tpr = recall_score(y_train_5, y_ppv_90)
logger.info('Recall score: {}'.format(tpr))

fpr, tpr, thresholds = roc_curve(y_train_5, y_score)

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()

auc = roc_auc_score(y_train_5, y_score)
logger.info('Area under curve: {}'.format(auc))

rnd_clf = RandomForestClassifier(random_state=42)
y_probas_score_rnd = cross_val_predict(rnd_clf, X_train, y_train_5, cv=3, method='predict_proba')
y_score_rnd = y_probas_score_rnd[:, 1]

auc_rnd = roc_auc_score(y_train_5, y_score_rnd)
logger.info('Area under curve: {}'.format(auc_rnd))

fpr_rnd, tpr_rnd, thresholds_rnd = roc_curve(y_train_5, y_score_rnd)

plt.plot(fpr, tpr, 'r:', label='SGD')
plt.plot(fpr_rnd, tpr_rnd, 'b:', label='RND')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC curve')
plt.axis([0, 1, 0, 1])
plt.legend()
plt.show()
