import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

history = pickle.load(open('history.pk', 'rb'))
logger.info(history.history.keys())

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = list(map(lambda x: x + 1, history.epoch))
epochs = range(1, len(history.epoch) + 1)

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

vai = np.argmax(val_acc)
vav = val_acc[vai]
logger.info('get maximum validation accuracy ({}) after {} epochs'.format(vav, vai))
vli = np.argmin(val_loss)
vlv = val_loss[vli]
logger.info('get minimum validation loss ({}) after {} epochs'.format(vlv, vli))
