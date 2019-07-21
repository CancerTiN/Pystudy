# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import numpy as np

X = np.array([6, 8, 10, 14, 18]).reshape(-1, 1)
print('explanatory variable: {}'.format(X))
x_bar = X.mean()
print('explanatory variable mean: {}'.format(x_bar))
variance = ((X - x_bar) ** 2).sum() / (X.shape[0] - 1)
print('explanatory variable variance: {}'.format(variance))
np_var = np.var(X, ddof=1)
print('explanatory variable variance: {}'.format(np_var))
y = np.array([7, 9, 13, 17.5, 18])
print('response variable: {}'.format(y))
y_bar = y.mean()
print('response variable mean: {}'.format(y_bar))
covariance = np.multiply((X - x_bar).transpose(), y - y_bar).sum() / (X.shape[0] - 1)
print('covariance: {}'.format(covariance))
beta = covariance / variance
print('beta: {}'.format(beta))
alpha = y_bar - beta * x_bar
print('alpha: {}'.format(alpha))

print(' test begin '.center(64, '#'))
func = lambda x: print('A {}" pizza should cost: ${:.2f}'.format(x, alpha + beta * x))
func(11)
func(18)
print(' test final '.center(64, '#'))
