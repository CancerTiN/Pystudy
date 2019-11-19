import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

history = pickle.load(open('history.pk', 'rb'))
logging.debug(history.history.keys())

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = list(map(lambda x: x + 1, history.epoch))
epochs = range(1, len(history.epoch) + 1)


def smooth_curve(points, factor=0.8):
    smoothed_points = list()
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


logger.info('Get max accuracy ({}) at epochs ({})'.format(max(val_acc), np.argmax(val_acc) + 1))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.waitforbuttonpress()
plt.close()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.waitforbuttonpress()
plt.close()

plt.plot(smooth_curve(epochs), acc, 'bo', label='Smoothed training acc')
plt.plot(smooth_curve(epochs), val_acc, 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.waitforbuttonpress()
plt.close()

plt.plot(smooth_curve(epochs), loss, 'bo', label='Smoothed training loss')
plt.plot(smooth_curve(epochs), val_loss, 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.waitforbuttonpress()
plt.close()
