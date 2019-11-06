import logging

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.applications import VGG16

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

model = VGG16(weights='imagenet', include_top=False)


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.0
    step = 1.0
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)


layer_name = 'block1_conv1'
size = 64
margin = 5

results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3)).astype(np.uint8)
logger.info(results.shape)

for i in range(8):
    for j in range(8):
        filter_index = i + (j * 8)
        filter_img = generate_pattern(layer_name, filter_index, size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        logger.info(((horizontal_start, horizontal_end), (vertical_start, vertical_end)))
        results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

plt.figure(figsize=(24, 24))
plt.imshow(results)
plt.waitforbuttonpress()
plt.close()
