import logging

import matplotlib.pyplot as plt
import numpy as np
from keras import models
from keras.preprocessing import image

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

model = models.load_model('cats_and_dogs_small_2.h5')
model.summary()

img_path = 'D:\\Workspace\\Study\\DLwP\\Ch05\\cat.1700.jpg'

img = image.load_img(img_path, target_size=(150, 150))
logger.info(img)
img_tensor = image.img_to_array(img)
logger.info(img_tensor)
logger.info(img_tensor.shape)
img_tensor = np.expand_dims(img_tensor, axis=0)
logger.info(img_tensor.shape)
img_tensor /= 255.0
logger.info(img_tensor.shape)
logger.info(img_tensor)

plt.imshow(img_tensor[0])
plt.waitforbuttonpress()
plt.close()

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
logger.info(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 4])
plt.waitforbuttonpress()
plt.close()
plt.matshow(first_layer_activation[0, :, :, 7])
plt.waitforbuttonpress()
plt.close()

layer_names = [layer.name for layer in model.layers[:8]]

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    logger.info(layer_name)
    logger.info(layer_activation.shape)

    size = layer_activation.shape[1]
    n_features = layer_activation.shape[-1]

    n_rows = n_features // images_per_row
    display_grid = np.zeros((size * n_rows, size * images_per_row))

    logger.info(n_rows)
    logger.info(display_grid.shape)

    for row in range(n_rows):
        for col in range(images_per_row):
            channel_image = layer_activation[0, :, :, row * images_per_row + col]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype(np.uint8)
            display_grid[row * size: (row + 1) * size, col * size: (col + 1) * size] = channel_image

    # scale = 1.0 / size
    # figsize = (scale * display_grid.shape[1], scale * display_grid.shape[0])
    # figsize = (images_per_row, n_rows)
    width = images_per_row
    height = n_rows
    figsize = (width, height)
    plt.figure(figsize=figsize)
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid)
    # plt.show()
    plt.waitforbuttonpress()
    plt.close()
