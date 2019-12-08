# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_breast_cancer

from sklearn.naive_bayes import GaussianNB

X = [[1.785e+01, 1.323e+01, 1.146e+02, 9.921e+02, 7.838e-02, 6.217e-02,
      4.445e-02, 4.178e-02, 1.220e-01, 5.243e-02, 4.834e-01, 1.046e+00,
      3.163e+00, 5.095e+01, 4.369e-03, 8.274e-03, 1.153e-02, 7.437e-03,
      1.302e-02, 1.309e-03, 1.982e+01, 1.842e+01, 1.271e+02, 1.210e+03,
      9.862e-02, 9.976e-02, 1.048e-01, 8.341e-02, 1.783e-01, 5.871e-02],
     [1.674e+01, 2.159e+01, 1.101e+02, 8.695e+02, 9.610e-02, 1.336e-01,
      1.348e-01, 6.018e-02, 1.896e-01, 5.656e-02, 4.615e-01, 9.197e-01,
      3.008e+00, 4.519e+01, 5.776e-03, 2.499e-02, 3.695e-02, 1.195e-02,
      2.789e-02, 2.665e-03, 2.001e+01, 2.902e+01, 1.335e+02, 1.229e+03,
      1.563e-01, 3.835e-01, 5.409e-01, 1.813e-01, 4.863e-01, 8.633e-02],
     [1.048e+01, 1.498e+01, 6.749e+01, 3.336e+02, 9.816e-02, 1.013e-01,
      6.335e-02, 2.218e-02, 1.925e-01, 6.915e-02, 3.276e-01, 1.127e+00,
      2.564e+00, 2.077e+01, 7.364e-03, 3.867e-02, 5.263e-02, 1.264e-02,
      2.161e-02, 4.830e-03, 1.213e+01, 2.157e+01, 8.141e+01, 4.404e+02,
      1.327e-01, 2.996e-01, 2.939e-01, 9.310e-02, 3.020e-01, 9.646e-02],
     [1.356e+01, 1.390e+01, 8.859e+01, 5.613e+02, 1.051e-01, 1.192e-01,
      7.860e-02, 4.451e-02, 1.962e-01, 6.303e-02, 2.569e-01, 4.981e-01,
      2.011e+00, 2.103e+01, 5.851e-03, 2.314e-02, 2.544e-02, 8.360e-03,
      1.842e-02, 2.918e-03, 1.498e+01, 1.713e+01, 1.011e+02, 6.866e+02,
      1.376e-01, 2.698e-01, 2.577e-01, 9.090e-02, 3.065e-01, 8.177e-02],
     [1.546e+01, 1.948e+01, 1.017e+02, 7.489e+02, 1.092e-01, 1.223e-01,
      1.466e-01, 8.087e-02, 1.931e-01, 5.796e-02, 4.743e-01, 7.859e-01,
      3.094e+00, 4.831e+01, 6.240e-03, 1.484e-02, 2.813e-02, 1.093e-02,
      1.397e-02, 2.461e-03, 1.926e+01, 2.600e+01, 1.249e+02, 1.156e+03,
      1.546e-01, 2.394e-01, 3.791e-01, 1.514e-01, 2.837e-01, 8.019e-02],
     [1.106e+01, 1.483e+01, 7.031e+01, 3.782e+02, 7.741e-02, 4.768e-02,
      2.712e-02, 7.246e-03, 1.535e-01, 6.214e-02, 1.855e-01, 6.881e-01,
      1.263e+00, 1.298e+01, 4.259e-03, 1.469e-02, 1.940e-02, 4.168e-03,
      1.191e-02, 3.537e-03, 1.268e+01, 2.035e+01, 8.079e+01, 4.967e+02,
      1.120e-01, 1.879e-01, 2.079e-01, 5.556e-02, 2.590e-01, 9.158e-02],
     [1.505e+01, 1.907e+01, 9.726e+01, 7.019e+02, 9.215e-02, 8.597e-02,
      7.486e-02, 4.335e-02, 1.561e-01, 5.915e-02, 3.860e-01, 1.198e+00,
      2.630e+00, 3.849e+01, 4.952e-03, 1.630e-02, 2.967e-02, 9.423e-03,
      1.152e-02, 1.718e-03, 1.758e+01, 2.806e+01, 1.138e+02, 9.670e+02,
      1.246e-01, 2.101e-01, 2.866e-01, 1.120e-01, 2.282e-01, 6.954e-02],
     [1.448e+01, 2.146e+01, 9.425e+01, 6.482e+02, 9.444e-02, 9.947e-02,
      1.204e-01, 4.938e-02, 2.075e-01, 5.636e-02, 4.204e-01, 2.220e+00,
      3.301e+00, 3.887e+01, 9.369e-03, 2.983e-02, 5.371e-02, 1.761e-02,
      2.418e-02, 3.249e-03, 1.621e+01, 2.925e+01, 1.084e+02, 8.089e+02,
      1.306e-01, 1.976e-01, 3.349e-01, 1.225e-01, 3.020e-01, 6.846e-02],
     [1.575e+01, 1.922e+01, 1.071e+02, 7.586e+02, 1.243e-01, 2.364e-01,
      2.914e-01, 1.242e-01, 2.375e-01, 7.603e-02, 5.204e-01, 1.324e+00,
      3.477e+00, 5.122e+01, 9.329e-03, 6.559e-02, 9.953e-02, 2.283e-02,
      5.543e-02, 7.330e-03, 1.736e+01, 2.417e+01, 1.194e+02, 9.153e+02,
      1.550e-01, 5.046e-01, 6.872e-01, 2.135e-01, 4.245e-01, 1.050e-01]]
y = [1, 0, 1, 1, 1, 0, 1, 0, 0]
model = GaussianNB()
model.fit(X, y)
label = model.predict([[1.795e+01, 2.001e+01, 1.142e+02, 9.820e+02, 8.402e-02, 6.722e-02,
                        7.293e-02, 5.596e-02, 2.129e-01, 5.025e-02, 5.506e-01, 1.214e+00,
                        3.357e+00, 5.404e+01, 4.024e-03, 8.422e-03, 2.291e-02, 9.863e-03,
                        5.014e-02, 1.902e-03, 2.058e+01, 2.783e+01, 1.292e+02, 1.261e+03,
                        1.072e-01, 1.202e-01, 2.249e-01, 1.185e-01, 4.882e-01, 6.111e-02]])
print('Lable: {}'.format(label))
# Lable: [0]

X = [['geneA', 'geneD', 'geneH', 'geneI', 'geneK', 'geneU', 'geneX', 'geneZ'],
     ['geneF', 'geneI', 'geneL', 'geneM', 'geneO', 'geneP', 'geneS', 'geneV'],
     ['geneC', 'geneF', 'geneH', 'geneI', 'geneN', 'geneO', 'geneQ', 'geneU'],
     ['geneA', 'geneE', 'geneF', 'geneG', 'geneH', 'geneI', 'geneP', 'geneV'],
     ['geneA', 'geneD', 'geneG', 'geneH', 'geneI', 'geneO', 'geneR', 'geneX'],
     ['geneA', 'geneD', 'geneG', 'geneH', 'geneI', 'geneO', 'geneR', 'geneX'],
     ['geneF', 'geneI', 'geneL', 'geneM', 'geneO', 'geneP', 'geneS', 'geneV'],
     ['geneC', 'geneE', 'geneK', 'geneP', 'geneR', 'geneS', 'geneX', 'geneY'],
     ['geneC', 'geneE', 'geneK', 'geneP', 'geneR', 'geneS', 'geneX', 'geneY']]
y = [1, 1, 0, 1, 0, 1, 0, 0, 1]
model = GaussianNB()
model.fit(X, y)
label1 = model.predict([['geneA', 'geneC', 'geneE', 'geneH', 'geneL', 'geneN', 'geneV', 'geneZ']])
label2 = model.predict([['geneC', 'geneG', 'geneI', 'geneL', 'geneP', 'geneS', 'geneV', 'geneY']])
label3 = model.predict([['geneA', 'geneC', 'geneE', 'geneF', 'geneH', 'geneM', 'geneP', 'geneV']])
print('Label1: {}; Label2: {}'.format(label1, label2))

word_set = set()

for doc in X:
    word_set.update(doc)
word_bag = sorted(list(word_set))

class0_pv = np.zeros((len(word_bag)))
class1_pv = np.zeros((len(word_bag)))
class0_num = class1_num = 0

for doc, label in zip(X, y):
    for word in doc:
        word_index = word_bag.index(word)
        {0: class0_pv, 1: class1_pv}[label][word_index] += 1
    if label == 0:
        class0_num += len(doc)
    elif label == 1:
        class1_num += len(doc)

class0_pv = class0_pv / class0_num
class1_pv = class1_pv / class1_num

sp_test = ['geneA', 'geneC', 'geneE', 'geneF', 'geneH', 'geneM', 'geneP', 'geneV']

sp_class0_pv = np.zeros((len(sp_test)))
sp_class1_pv = np.zeros((len(sp_test)))
word_bag = sorted(word_bag)

for i, w in enumerate(sp_test):
    w_index = word_bag.index(w)
    sp_class0_pv[i] = class0_pv[w_index]
    sp_class1_pv[i] = class1_pv[w_index]

_X, _y = load_breast_cancer(return_X_y=True)

_X[:10]
