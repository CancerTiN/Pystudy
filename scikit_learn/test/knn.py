# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

from sklearn.neighbors import KNeighborsClassifier
X = [[30, 20], [40, 20], [10, 30], [15, 25], [20, 25]]
y = [0, 0, 1, 1, 1]
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
label1 = neigh.predict([[25, 25]])
label2 = neigh.predict([[20, 20]])
print('Label1: {}; Label2: {}'.format(label1, label2))
