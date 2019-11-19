import logging
import pickle

from keras.datasets import imdb
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing import sequence

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

max_features = 10000
maxlen = 500
batch_size = 32

logger.info('start loading data')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
logger.info('succeed in loading {} train sequences'.format(len(x_train)))
logger.info('succeed in loading {} test sequences'.format(len(x_test)))

logger.info('start padding sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
logger.info('succeed in getting x_train with shape ({}, {})'.format(*x_train.shape))
logger.info('succeed in getting x_test with shape ({}, {})'.format(*x_test.shape))

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)
pickle.dump(history, open('history.pk', 'wb'))
pickle.dump(model, open('model.pk', 'wb'))
