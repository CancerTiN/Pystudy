# -*- coding: utf-8 -*-

import logging

import numpy as np
from sklearn import metrics

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_conditional_entropy(xs, ys):
    hxy = 0.0
    xy_tuple = tuple(zip(xs, ys))
    y_tuple = tuple(ys[:len(xy_tuple)])
    xy_set = set(xy_tuple)
    n = len(xy_tuple)
    for xy in xy_set:
        nxy = xy_tuple.count(xy)
        ny = y_tuple.count(xy[-1])
        hxy -= nxy / n * np.log(nxy / ny)
    return hxy


def get_entropy(xs):
    hx = 0.0
    x_tuple = tuple(xs)
    x_set = set(x_tuple)
    n = len(x_tuple)
    for x in x_set:
        nx = x_tuple.count(x)
        hx -= nx / n * np.log(nx / n)
    return hx


def get_ari(y_true, y_pred):
    ri


data = (([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2]),
        ([0, 0, 0, 1, 1, 1], [0, 0, 1, 3, 3, 3]),
        ([0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0]))

for y_true, y_pred in data:
    logger.info('C: {}'.format(y_true))
    logger.info('K: {}'.format(y_pred))

    h_score_self = 1 - get_conditional_entropy(y_true, y_pred) / get_entropy(y_true)
    c_score_self = 1 - get_conditional_entropy(y_pred, y_true) / get_entropy(y_pred)
    v_score_self = 2 * h_score_self * c_score_self / (h_score_self + c_score_self)

    h_score = metrics.homogeneity_score(y_true, y_pred)
    c_score = metrics.completeness_score(y_true, y_pred)
    v_score = metrics.v_measure_score(y_true, y_pred)

    assert round(h_score_self) == round(h_score)
    logger.info('Homogeneity score: {}'.format(h_score))

    assert round(c_score_self) == round(c_score)
    logger.info('Completeness score: {}'.format(c_score))

    assert round(v_score_self) == round(v_score)
    logger.info('V measure score: {}'.format(v_score))

    ari = metrics.adjusted_rand_score(y_true, y_pred)
    logger.info('Adjusted Rand Score: {}'.format(ari))
