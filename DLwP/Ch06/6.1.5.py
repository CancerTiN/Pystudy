import logging
import pickle

from keras import layers
from keras import models
from keras import preprocessing
from keras.datasets import imdb

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

num_words = 10000
maxlen = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

logger.info(x_train.shape)
logger.info(x_test.shape)
logger.info([len(x) for x in x_train[:10]])
logger.info([len(x) for x in x_test[:10]])
logger.info(y_train)
logger.info(y_test)

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

logger.info(x_train.shape)
logger.info(x_test.shape)

model = models.Sequential()
model.add(layers.Embedding(10000, 8, input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2)
pickle.dump(history, open('history.pk', 'wb'))


