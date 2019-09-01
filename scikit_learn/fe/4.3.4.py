# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

from nltk import pos_tag
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

corpus = [
    'He ate the sandwiches',
    'Every sandwich was eaten by him'
]

stemmer = PorterStemmer()
stemmed = [[stemmer.stem(token) for token in word_tokenize(document)] for document in corpus]

lemmatizer = WordNetLemmatizer()


def lemmatize(token, tag):
    if tag[0].lower() in list('nv'):
        return lemmatizer.lemmatize(token, tag[0].lower())
    return token


lemmatized = [[lemmatize(token, tag) for token, tag in document] for document in
              [pos_tag(word_tokenize(document)) for document in corpus]]

print('stemmed:\n{}'.format(stemmed))
print('lemmatized:\n{}'.format(lemmatized))
