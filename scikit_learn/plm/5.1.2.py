# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import numpy as np

from sklearn.linear_model import LinearRegression

X = np.array([[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]])
y = np.array([7, 9, 13, 17.5, 18]).reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)

X_test = np.array([[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]])
y_test = np.array([11, 8.5, 15, 18, 11]).reshape(-1, 1)

predictions = model.predict(X_test)
for i, pred in enumerate(predictions):
    print('Predicted: {}; Target: {}'.format(pred, y_test[i]))
else:
    print('R-squared: {}'.format(model.score(X_test, y_test)))
