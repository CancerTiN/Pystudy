import logging

import numpy as np
from keras import layers
from keras import models
from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

logging.debug(train_data.shape)
logging.debug(test_data.shape)
logging.debug(train_targets)

scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(13,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = list()

for i in range(k):
    logging.debug('processing hold #{}'.format(i))
    start = i * num_val_samples
    end = (i + 1) * num_val_samples
    val_data = train_data[start: end]
    val_targets = train_targets[start: end]
    partial_train_data = np.concatenate([train_data[:start], train_data[end:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:start], train_targets[end:]], axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets, batch_size=1, epochs=num_epochs, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    logging.debug(val_mse)
    all_scores.append(val_mae)

logging.debug(all_scores)
logging.debug(np.mean(all_scores))
