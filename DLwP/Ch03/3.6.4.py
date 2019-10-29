import logging

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


model = build_model()
model.fit(train_data, train_targets, batch_size=16, epochs=43)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

logging.debug(test_mse_score)
logging.debug(test_mae_score)
