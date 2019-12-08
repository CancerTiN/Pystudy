# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier

X = [[1, 1], [1, 1], [1, 0], [0, 1], [0, 1]]
y = [0, 0, 1, 1, 1]
model = DecisionTreeClassifier()
model.fit(X, y)
label1 = model.predict([[1, 1]])
label2 = model.predict([[0, 0]])
print('Label1: {}; Label2: {}'.format(label1, label2))
