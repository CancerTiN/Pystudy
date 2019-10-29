import pickle

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'D:\\Workspace\\Study\\DLwP\\Ch05\\train'
train_cats_dir = 'D:\\Workspace\\Study\\DLwP\\Ch05\\train\\cats'
train_dogs_dir = 'D:\\Workspace\\Study\\DLwP\\Ch05\\train\\dogs'
validation_dir = 'D:\\Workspace\\Study\\DLwP\\Ch05\\validation'
validation_cats_dir = 'D:\\Workspace\\Study\\DLwP\\Ch05\\validation\\cats'
validation_dogs_dir = 'D:\\Workspace\\Study\\DLwP\\Ch05\\validation\\dogs'
test_dir = 'D:\\Workspace\\Study\\DLwP\\Ch05\\test'
test_cats_dir = 'D:\\Workspace\\Study\\DLwP\\Ch05\\test\\cats'
test_dogs_dir = 'D:\\Workspace\\Study\\DLwP\\Ch05\\test\\dogs'

train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32,
                                                    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=32,
                                                              class_mode='binary')

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator,
                              validation_steps=50)
model.save('cats_and_dogs_small_2.h5')
pickle.dump(history, open('history.pk', 'wb'))

