import logging

import numpy as np
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype(np.float32) / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype(np.float32) / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

for arr in (train_images, test_images, train_labels, test_labels):
    logging.debug(arr[0])

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=64, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
logging.debug(test_loss)
logging.debug(test_acc)
