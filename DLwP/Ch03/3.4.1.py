import logging
import pickle

import numpy as np
from keras import layers
from keras import losses
from keras import metrics
from keras import models
from keras import optimizers
from keras.datasets import imdb

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

logging.debug(train_data[0])
logging.debug(train_labels[0])
logging.debug(max(max(sequence) for sequence in train_data))

word_index = imdb.get_word_index()
reverse_word_index = dict((value, key) for key, value in word_index.items())
decoded_review = ' '.join(reverse_word_index.get(i - 3, '?') for i in train_data[0])

logging.debug(decoded_review)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

logging.debug(x_train[0])

y_train = np.asarray(train_labels).astype(np.float32)
y_test = np.asarray(test_labels).astype(np.float32)

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

flag = False
if flag:
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizers.RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizers.RMSprop(), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(partial_x_train, partial_y_train, batch_size=512, epochs=20, validation_data=(x_val, y_val))

pickle.dump(history, open('history.pk', 'wb'))
