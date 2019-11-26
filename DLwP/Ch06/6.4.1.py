import logging
import os
import pickle

os.environ['KERAS_BACKEND'] = 'tensorflow'
logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from keras import layers
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.optimizers import RMSprop

max_features = 10000
maxlen = 500

logger.debug('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
logger.debug('Receive {} train sequences.'.format(len(x_train)))
logger.debug('Receive {} test sequences.'.format(len(x_test)))

logger.debug('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
logger.debug('Shape of x_train is {}'.format(x_train.shape))
logger.debug('Shape of x_test is {}'.format(x_test.shape))

model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=maxlen))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

pickle.dump(history, open('history.641.pk', 'wb'))
