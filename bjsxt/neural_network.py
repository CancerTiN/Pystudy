# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

X = np.array([[0, 0], [1, 1]])
y = np.array([0, 1]).reshape(-1, 1)

clf = MLPClassifier(hidden_layer_sizes=(5, 2), activation='logistic', solver='sgd',
                    alpha=1e-5, max_iter=2000, tol=1e-4)
clf.fit(X, y)

coef = clf.coefs_

X_test = np.array([[2, 2], [-1, -2]])

y_predict = clf.predict(X_test)
y_pred_proba = clf.predict(X_test)

logger.info('Loading data...')

df = pd.read_csv('concrete.csv')

X_indics = 'cement,slag,ash,water,superplastic,coarseagg,fineagg,age'.split(',')
X = df.reindex(X_indics, axis=1)
y = df['strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logger.info('Training model...')

reg = MLPRegressor()
reg.fit(X_train, y_train)

logger.info('Testing model...')

y_pred = reg.predict(X_test)
evs = explained_variance_score(y_test, y_pred)
aae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = median_absolute_error(y_test, y_pred)
r2s = r2_score(y_test, y_pred)

logger.info('Explained Variance Score: {}'.format(evs))
logger.info('Mean Absolute Error: {}'.format(aae))
logger.info('Mean Squared Error: {}'.format(mse))
logger.info('Median Absolute Error: {}'.format(mae))
logger.info('R2 Score: {}'.format(r2s))
