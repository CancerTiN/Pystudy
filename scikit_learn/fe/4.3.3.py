# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

corpus = [
    'I am gathering ingredients for the sandwich',
    'There were many wizards at the gathering'
]

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
for document in corpus:
    for token, tag in pos_tag(word_tokenize(document)):
        print(token)
        print(tag)
        print('{} -> {}'.format(token, stemmer.stem(token)))
        if tag[0].lower() in list('nv'):
            print('{} -> {}'.format(token, lemmatizer.lemmatize(token, tag[0].lower())))
        else:
            print('{} -> {}'.format(token, token))

