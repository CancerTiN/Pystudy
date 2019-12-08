# -*- coding: utf-8 -*-

from sklearn.ensemble import AdaBoostClassifier

X = [[10, 21], [20, 11], [13, 10], [10, 10], [20, 10]]
y = [1, 1, -1, -1, 1]
model = AdaBoostClassifier()
model.fit(X, y)
label1, proba1 = model.predict([[50, 50]]), model.predict_proba([[50, 50]])
label2, proba2 = model.predict([[0, 0]]), model.predict_proba([[0, 0]])
print('Label1: {}; Label2: {}'.format(label1, label2))
print('Proba1: {}; Proba2: {}'.format(proba1, proba2))
