import logging
import pickle

from keras import layers
from keras import models
from keras import optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

train_dir = 'D:\\Workspace\\Study\\DLwP\\Ch05\\train'
validation_dir = 'D:\\Workspace\\Study\\DLwP\\Ch05\\validation'
test_dir = 'D:\\Workspace\\Study\\DLwP\\Ch05\\test'

train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32,
                                                    class_mode='binary')
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=32,
                                                              class_mode='binary')

conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

logging.info(
    'this is the number of trainable weights before freezing the conv base: {}'.format(len(model.trainable_weights)))
conv_base.trainable = False
logging.info(
    'this is the number of trainable weights after freezing the conv base: {}'.format(len(model.trainable_weights)))

model.summary()

conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.summary()

model.compile(optimizer=optimizers.RMSprop(2e-5), loss='binary_crossentropy', metrics=['acc'])

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator,
                              validation_steps=50)
pickle.dump(history, open('history.pk', 'wb'))
pickle.dump(model, open('model.pk', 'wb'))
