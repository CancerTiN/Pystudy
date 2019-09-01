# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english')
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]

X_csr_matrix = vectorizer.fit_transform(corpus)
X = X_csr_matrix.todense()
vectorizer_vocabulary = vectorizer.vocabulary_
print('vectorizer fit transform return:\n{}'.format(X_csr_matrix))
print('return to dense:\n{}'.format(X))
print('vectorizer vocabulary:\n{}'.format(vectorizer_vocabulary))
