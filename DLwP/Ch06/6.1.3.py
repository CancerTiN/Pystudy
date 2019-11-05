import logging

from keras.preprocessing.text import Tokenizer

logging.basicConfig(format='%(asctime)s\t%(name)s\t%(levelname)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)
logger.info(sequences)
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
logger.info(one_hot_results)
logger.info(one_hot_results.shape)

word_index = tokenizer.word_index
logger.info('found {} unique tokens\n{}'.format(len(word_index), word_index))
