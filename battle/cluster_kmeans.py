# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples = 400
n_features = 2
centers = 4

data1, y1 = make_blobs(n_samples, n_features, centers, random_state=2)
data2, y2 = make_blobs(n_samples, n_features, centers, cluster_std=(1, 2.5, 0.5, 2), random_state=2)
data3 = np.vstack((data1[y1 == 0], data1[y1 == 1][:50], data1[y1 == 2][:20], data1[y1 == 3][:5]))
y3 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5)

cls = KMeans(n_clusters=4)

y1_hat = cls.fit_predict(data1)
y2_hat = cls.fit_predict(data2)
y3_hat = cls.fit_predict(data3)

data_r = np.dot(data1, np.array(((1, 1), (1, 3))))
y_r_hat = cls.fit_predict(data_r)

cmap = matplotlib.colors.ListedColormap('rgbm')


def expand(_min, _max):
    _dis = (_max - _min) * 0.1
    return _min - _dis, _max + _dis


plt.figure(figsize=(9, 10), facecolor='w')
plt.subplot(421)
plt.title('Raw data')
plt.scatter(data1[:, 0], data1[:, 1], c=y1, cmap=cmap)
x1_min, x2_min = np.min(data1, axis=0)
x1_max, x2_max = np.max(data1, axis=0)
left, right = expand(x1_min, x1_max)
bottom, top = expand(x2_min, x2_max)
plt.xlim((left, right))
plt.ylim((bottom, top))
plt.grid(True)

plt.subplot(422)
plt.title('K-means++ data')
plt.scatter(data1[:, 0], data1[:, 1], c=y1_hat, cmap=cmap)
x1_min, x2_min = np.min(data1, axis=0)
x1_max, x2_max = np.max(data1, axis=0)
left, right = expand(x1_min, x1_max)
bottom, top = expand(x2_min, x2_max)
plt.xlim((left, right))
plt.ylim((bottom, top))
plt.grid(True)

plt.subplot(423)
plt.title('Rotated raw data')
plt.scatter(data_r[:, 0], data_r[:, 1], c=y1, cmap=cmap)
x1_min, x2_min = np.min(data_r, axis=0)
x1_max, x2_max = np.max(data_r, axis=0)
left, right = expand(x1_min, x1_max)
bottom, top = expand(x2_min, x2_max)
plt.xlim((left, right))
plt.ylim((bottom, top))
plt.grid(True)

plt.subplot(424)
plt.title('Rotated k-means++ data')
plt.scatter(data_r[:, 0], data_r[:, 1], c=y_r_hat, cmap=cmap)
x1_min, x2_min = np.min(data_r, axis=0)
x1_max, x2_max = np.max(data_r, axis=0)
left, right = expand(x1_min, x1_max)
bottom, top = expand(x2_min, x2_max)
plt.xlim((left, right))
plt.ylim((bottom, top))
plt.grid(True)

plt.subplot(425)
plt.title('Uncertain variance data')
plt.scatter(data2[:, 0], data2[:, 1], c=y2, cmap=cmap)
x1_min, x2_min = np.min(data2, axis=0)
x1_max, x2_max = np.max(data2, axis=0)
left, right = expand(x1_min, x1_max)
bottom, top = expand(x2_min, x2_max)
plt.xlim((left, right))
plt.ylim((bottom, top))
plt.grid(True)

plt.subplot(426)
plt.title('Uncertain variance k-means++ data')
plt.scatter(data2[:, 0], data2[:, 1], c=y2_hat, cmap=cmap)
x1_min, x2_min = np.min(data2, axis=0)
x1_max, x2_max = np.max(data2, axis=0)
left, right = expand(x1_min, x1_max)
bottom, top = expand(x2_min, x2_max)
plt.xlim((left, right))
plt.ylim((bottom, top))
plt.grid(True)

plt.subplot(427)
plt.title('Uneven data')
plt.scatter(data3[:, 0], data3[:, 1], c=y3, cmap=cmap)
x1_min, x2_min = np.min(data3, axis=0)
x1_max, x2_max = np.max(data3, axis=0)
left, right = expand(x1_min, x1_max)
bottom, top = expand(x2_min, x2_max)
plt.xlim((left, right))
plt.ylim((bottom, top))
plt.grid(True)

plt.subplot(428)
plt.title('Uneven k-means++ data')
plt.scatter(data3[:, 0], data3[:, 1], c=y3_hat, cmap=cmap)
x1_min, x2_min = np.min(data3, axis=0)
x1_max, x2_max = np.max(data3, axis=0)
left, right = expand(x1_min, x1_max)
bottom, top = expand(x2_min, x2_max)
plt.xlim((left, right))
plt.ylim((bottom, top))
plt.grid(True)

plt.suptitle('Impact of data distribution on K-means clustering', fontsize=18)
plt.tight_layout(2, rect=(0, 0, 1, 0.97))
plt.show()
