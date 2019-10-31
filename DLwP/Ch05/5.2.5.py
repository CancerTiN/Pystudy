import logging
import pickle

import numpy as np
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
conv_base.summary()

train_dir = 'D:\\Workspace\\Study\\DLwP\\Ch05\\train'
validation_dir = 'D:\\Workspace\\Study\\DLwP\\Ch05\\validation'
test_dir = 'D:\\Workspace\\Study\\DLwP\\Ch05\\test'

datagen = ImageDataGenerator(rescale=1.0 / 255)
batch_size = 20


def extract_features(directory, sample_count):
    features = np.zeros((sample_count, 4, 4, 512))
    lables = np.zeros((sample_count))
    generator = datagen.flow_from_directory(directory, target_size=(150, 150), class_mode='binary',
                                            batch_size=batch_size)
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        start = i * batch_size
        end = (i + 1) * batch_size
        features[start:end] = features_batch
        lables[start:end] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, lables


train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

extracted_objects = [train_features, train_labels,
                     validation_features, validation_labels,
                     test_features, test_labels]
pickle.dump(extracted_objects, open('extracted_objects.pk', 'wb'))
