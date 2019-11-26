import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if len(sys.argv) < 2:
    logger.error('Missing argument in command, abord')
    sys.exit(-1)
else:
    history_pk = sys.argv[1]
    if not os.path.isfile(history_pk):
        logger.error('Can not find {}, abord'.format(history_pk))
        sys.exit(-2)
    if len(sys.argv) >= 3:
        if sys.argv[2] == '-s':
            flag_smooth = True
        else:
            flag_smooth = False
    else:
        flag_smooth = False

history = pickle.load(open(sys.argv[1], 'rb'))
logging.info(history.history.keys())

loss = history.history['loss']
val_loss = history.history['val_loss']
if 'acc' in history.history:
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    flag_acc = True
else:
    flag_acc = False

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


if flag_acc:
    logger.info('Get max accuracy ({}) at epochs ({})'.format(max(val_acc), np.argmax(val_acc) + 1))

if flag_acc:
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.waitforbuttonpress()
    plt.close()

logger.info('Get min loss ({}) at epochs ({})'.format(max(val_loss), np.argmin(val_loss) + 1))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.waitforbuttonpress()
plt.close()

if flag_smooth:
    if flag_acc:
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
