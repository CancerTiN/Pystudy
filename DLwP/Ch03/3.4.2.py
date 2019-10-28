import logging
import pickle

import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

history = pickle.load(open('history.pk', 'rb'))
history_dict = history.history
logging.debug(history_dict.keys())

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']

epochs = list(map(lambda x: x + 1, history.epoch))

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.waitforbuttonpress()
plt.close()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.waitforbuttonpress()
plt.close()
