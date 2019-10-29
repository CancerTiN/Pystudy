import logging
import pickle

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
num_epochs = 500
all_mae_histories = list()

for i in range(k):
    logging.debug('processing fold #{}'.format(i))
    start = i * num_val_samples
    end = (i + 1) * num_val_samples
    val_data = train_data[start: end]
    val_targets = train_targets[start: end]
    partial_train_data = np.concatenate([train_data[:start], train_data[end:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:start], train_targets[end:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, batch_size=1, epochs=num_epochs, verbose=0,
                        validation_data=(val_data, val_targets))
    mae_history = history.history['val_mean_absolute_error']
    logging.debug(mae_history)
    all_mae_histories.append(mae_history)
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

logging.debug(all_mae_histories)
logging.debug(average_mae_history)

pickle.dump(average_mae_history, open('average_mae_history.pk', 'wb'))
