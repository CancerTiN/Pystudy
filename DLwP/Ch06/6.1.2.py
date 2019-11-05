import logging
import string

import numpy as np

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable

token_index = dict(zip(characters, range(1, len(characters) + 1)))
logger.info(token_index)

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, character in list(enumerate(sample))[:max_length]:
        index = token_index.get(character)
        results[i, j, index] = 1
logger.info(results)

index_token = {v: k for k, v in token_index.items()}
for result in results:
    ret = np.nonzero(result)
    chars = list()
    for index in ret[1]:
        token = index_token.get(index)
        chars.append(token)
    logger.info(''.join(chars))
