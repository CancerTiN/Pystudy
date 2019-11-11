import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

img_path = r'D:\Workspace\Study\DLwP\Ch05\creative_commons_elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

model = VGG16(weights='imagenet')
preds = model.predict(x)
logger.info(preds)
results = decode_predictions(preds, top=3)
logger.info(results)
logger.info(np.argmax(preds))

african_elephant_output = model.output[:, 386]
last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.waitforbuttonpress()
plt.close()

img = cv2.imread(img_path)
width = img.shape[1]
height = img.shape[0]
heatmap = cv2.resize(heatmap, (width, height))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = img + heatmap * 0.4
cv2.imwrite('elephant_cam.jpg', superimposed_img)
