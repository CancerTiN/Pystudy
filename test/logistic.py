# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
X = [[30, 20], [40, 20], [10, 30], [15, 25], [20, 25]]
y = [0, 0, 1, 1, 1]
model = LogisticRegression(solver='lbfgs')
model.fit(X, y)
label1, proba1 = model.predict([[25, 25]]), model.predict_proba([[25, 25]])
label2, proba2 = model.predict([[25, 20]]), model.predict_proba([[25, 20]])
print('Label1: {}; Label2: {}'.format(label1, label2))
print('Proba1: {}; Proba2: {}'.format(proba1, proba2))
