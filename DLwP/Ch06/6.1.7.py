import logging
import os

import numpy as np
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

imdb_dir = 'D:\\Workspace\\Study\\DLwP\\Ch06\\aclImdb'
test_dir = os.path.join(imdb_dir, 'test')
texts = list()
labels = list()
for label_type in ('neg', 'pos'):
    dir_name = os.path.join(test_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            texts.append(open(os.path.join(dir_name, fname), encoding='utf-8').read())
            labels.append(0 if label_type == 'neg' else 1)
logger.info('found {} texts'.format(len(texts)))
logger.info('found {} labels'.format(len(labels)))

maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000
embedding_dim = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.load_weights('pre_trained_glove_model.h5')
model.summary()
test_loss, test_acc = model.evaluate(x_test, y_test)
logger.info(test_loss)
logger.info(test_acc)
