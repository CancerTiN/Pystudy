# -*- coding: utf-8 -*-

import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def expand(_min, _max):
    _dis = (_max - _min) * 0.1
    return _min - _dis, _max + _dis


n_samples = 1000
n_features = 2
centers = ([1, 2], [-1, -1], [1, -1], [-1, 1])
cluster_std = (0.5, 0.25, 0.7, 0.5)

data, y_true = make_blobs(n_samples, n_features, centers, cluster_std, random_state=0)
X = StandardScaler().fit_transform(data)

params = ((0.2, 5), (0.2, 10), (0.2, 15), (0.3, 5), (0.3, 10), (0.3, 15))

for i in range(len(params)):
    eps, min_samples = params[i]
    model = DBSCAN(eps, min_samples)
    model.fit(data)
    y_pred = model.labels_

    core_indices = np.zeros_like(y_pred, dtype=bool)
    core_indices[model.core_sample_indices_] = True

    y_uniq = np.unique(y_pred)
    n_clusters = y_uniq.size - (1 if -1 in y_pred else 0)

    logger.info('Cluster number: {}'.format(n_clusters))

    x1_min, x2_min = np.min(data, axis=0)
    x1_max, x2_max = np.max(data, axis=0)
    left, right = expand(x1_min, x1_max)
    bottom, top = expand(x2_min, x2_max)

    plt.subplot(2, 3, i + 1)
    plt.xlim((left, right))
    plt.ylim((bottom, top))
    plt.grid(True)
    plt.title('eps = {} min_samples = {} n_clusters = {}'.format(eps, min_samples, n_clusters))

    colors = plt.cm.Spectral(np.linspace(0, 0.8, y_uniq.size))
    for k, c in zip(y_uniq, colors):
        k_indices = (y_pred == k)
        if k == -1:
            plt.scatter(data[k_indices, 0], data[k_indices, 1], s=20, c='k')
        else:
            plt.scatter(data[k_indices, 0], data[k_indices, 1], s=30, c=c)
            plt.scatter(data[k_indices & core_indices, 0], data[k_indices & core_indices, 1], s=40, c=c, edgecolors='k')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
