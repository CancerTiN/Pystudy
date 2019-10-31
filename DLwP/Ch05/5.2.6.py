import logging
import pickle

import numpy as np
from keras import layers
from keras import models
from keras import optimizers

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

train_features, train_labels, validation_features, validation_labels, test_features, test_labels = \
    pickle.load(open('extracted_objects.pk', 'rb'))

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

logging.info(train_features)
logging.info(train_labels)

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(2e-5), loss='binary_crossentropy', metrics=['acc'])

history = model.fit(train_features, train_labels, batch_size=20, epochs=30,
                    validation_data=(validation_features, validation_labels))
pickle.dump(history, open('history.pk', 'wb'))
