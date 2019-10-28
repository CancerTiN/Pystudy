import copy
import logging
import time

import numpy as np
from keras import layers
from keras import models
from keras.datasets import reuters

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

logging.debug(len(train_data))
logging.debug(len(test_data))
logging.debug(train_data[10])

word_index = reuters.get_word_index()
reverse_word_index = dict((value, key) for key, value in word_index.items())
decoded_newswire = ' '.join(reverse_word_index.get(i - 3, '?') for i in train_data[0])

logging.debug(decoded_newswire)
logging.debug(train_labels[10])


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.array(train_labels)
y_test = np.array(test_labels)

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = model.fit(partial_x_train, partial_y_train, batch_size=512, epochs=20, validation_data=(x_val, y_val))
results = model.evaluate(x_test, y_test)

time.sleep(1)
logging.debug(['categorical_crossentropy', 'accuracy'])
logging.debug(results)

test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
logging.debug(np.sum(hits_array) / len(test_labels))

predictions = model.predict(x_test)
logging.debug(predictions[0].shape)
logging.debug(np.sum(predictions[0]))
logging.debug(np.argmax(predictions[0]))
