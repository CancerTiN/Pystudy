# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
from scipy import fft
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.info('Loading data...')

X = list()
y = list()

categories = ['classical', 'country', 'jazz', 'metal', 'pop', 'rock']

for fname in os.listdir('trainset'):
    fpath = os.path.join('trainset', fname)
    if os.path.isdir(fpath):
        continue
    fft_features = np.load(fpath)
    category = fname.split('.')[0]
    category_index = categories.index(category)
    X.append(fft_features)
    y.append(category_index)

X = np.array(X)
y = np.array(y)

logger.info('Training model...')

model = LogisticRegression()
model.fit(X, y)

logger.info('Read test wavfile...')
fs, data = wavfile.read('trainset/sample/heibao-wudizirong-remix.wav')
x_test = abs(fft(data))[:1000]

logger.info('Test model...')
category_index = model.predict([x_test])[0]

logger.info('Test sample is classified as {}'.format(categories[category_index]))
