import logging

import numpy as np

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

timestep = 100
input_features = 32
output_features = 64

inputs = np.random.random((timestep, input_features))
logger.info('inputs: {}'.format(inputs))
state_t = np.zeros((output_features,))
logger.info('state_t: {}'.format(state_t))

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))
logger.info('W: {}'.format(W))
logger.info('U: {}'.format(U))
logger.info('b: {}'.format(b))

successive_outputs = list()
for input_t in inputs:
    x = np.dot(W, input_t) + np.dot(U, state_t) + b
    output_t = np.tanh(x)
    successive_outputs.append(output_t)
    state_t = output_t
    logger.info('x: {}'.format(x))
    logger.info('output_t: {}'.format(output_t))

final_output_sequence = np.stack(successive_outputs, axis=0)
logger.info('final_output_sequence: {}'.format(final_output_sequence))
