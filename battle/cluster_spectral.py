# -*- coding: utf-8 -*-

import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import spectral_clustering
from sklearn.metrics import euclidean_distances

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def expand(_min, _max):
    _dis = (_max - _min) * 0.1
    return _min - _dis, _max + _dis


x = np.arange(0, 2 * np.pi, 0.1)
data1 = np.vstack((1 * np.cos(x), 1 * np.sin(x))).T
data2 = np.vstack((2 * np.cos(x), 2 * np.sin(x))).T
data3 = np.vstack((3 * np.cos(x), 3 * np.sin(x))).T
data = np.vstack((data1, data2, data3))

x1_min, x2_min = np.min(data, axis=0)
x1_max, x2_max = np.max(data, axis=0)
left, right = expand(x1_min, x1_max)
bottom, top = expand(x2_min, x2_max)

n_clusters = 3

distances = euclidean_distances(data, squared=True)
_sigma = np.median(distances)

colors = plt.cm.Spectral(np.linspace(0, 0.8, n_clusters))

plt.figure(figsize=(12, 8))
plt.suptitle('Spectral cluster')

for i, s in enumerate(np.logspace(-2, 0, 6)):
    affinity = np.exp(-distances ** 2 / (s ** 2)) + 1e-6
    y_pred = spectral_clustering(affinity, n_clusters)

    plt.subplot(2, 3, i + 1)
    for k, c in enumerate(colors):
        k_indices = (y_pred == k)
        plt.scatter(data[k_indices, 0], data[k_indices, 1], c=c, edgecolors='k')
    plt.xlim((left, right))
    plt.ylim((bottom, top))
    plt.grid(True)
    plt.title('sigma = {}'.format(s))

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
