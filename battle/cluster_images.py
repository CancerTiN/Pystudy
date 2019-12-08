# -*- coding: utf-8 -*-

import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def show_scatter(sample):
    N = 10
    hist, edges = np.histogramdd(sample, bins=[N] * sample.shape[-1], range=[(0, 1)] * sample.shape[-1])
    hist /= hist.max()
    x = y = z = np.arange(N)
    xs, ys, zs = np.meshgrid(x, y, z)

    Axes3D
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c='r', s=100 * hist)
    ax.set_xlabel('R component')
    ax.set_ylabel('G component')
    ax.set_zlabel('B component')
    plt.title('3D frequency distribution of image color', fontsize=20)

    density = hist[hist > 0]
    density = np.sort(density)[::-1]
    frequency = np.arange((len(density)))

    plt.figure(2)
    plt.plot(frequency, density, 'r-', frequency, density, 'go', lw=2)
    plt.title('Image color frequency distribution')
    plt.grid(True)

    plt.show()


def restore_image(cluster_centers, cluster_results, image_shape):
    n_rows, n_cols, _ = image_shape
    image = np.empty((n_rows, n_cols, 3))
    i = 0
    for r in range(n_rows):
        for c in range(n_cols):
            image[r, c] = cluster_centers[cluster_results[i]]
            i += 1
    return image


im = Image.open('Lena.png')
image = np.array(im).astype(float) / 255
image_v = image.reshape((-1, 3))

show_scatter(image_v)

size = 1000
indices = np.random.randint(0, image_v.shape[0], size=size)
X = image_v[indices]

n_clusters = 2
model = KMeans(n_clusters)
model.fit(X)
y_pred = model.predict(image_v)

logger.info('Cluster results: {}'.format(y_pred))
logger.info('Cluster centers: {}'.format(model.cluster_centers_))

image_vq = restore_image(model.cluster_centers_, y_pred, image.shape)

plt.figure(figsize=(16, 8))

plt.subplot(121)
plt.axis('off')
plt.title('Raw image', fontsize=16)
plt.imshow(image)

plt.subplot(122)
plt.axis('off')
plt.title('Vector quantified image with {} colors'.format(n_clusters), fontsize=16)
plt.imshow(image_vq)

plt.tight_layout(3.0)
plt.show()
