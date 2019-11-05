import logging
import os
import pickle

import numpy as np
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

imdb_dir = 'D:\\Workspace\\Study\\DLwP\\Ch06\\aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
texts = list()
labels = list()
for label_type in ('neg', 'pos'):
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            texts.append(open(os.path.join(dir_name, fname), encoding='utf-8').read())
            labels.append(0 if label_type == 'neg' else 1)
logger.info('found {} texts'.format(len(texts)))
logger.info('found {} labels'.format(len(labels)))

glove_dir = 'D:\\Workspace\\Study\\DLwP\\Ch06'
embeddings_index = dict()
for line in open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf-8'):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype=np.float32)
    embeddings_index[word] = coefs
logger.info('found {} word vectors'.format(len(embeddings_index)))

maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
logger.info('found {} unique tokens\n{}'.format(len(word_index), word_index))

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, index in word_index.items():
    if index < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
logger.info('shape of data tensor is {}'.format(data.shape))
logger.info('shape of labels tensor is {}'.format(labels.shape))

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

use_embedding_matrix = False
if use_embedding_matrix:
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False
    model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
pickle.dump(history, open('history.pk', 'wb'))
model.save_weights('pre_trained_glove_model.h5')
