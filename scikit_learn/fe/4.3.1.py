# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances

vectorizer = CountVectorizer()
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
]

X_csr_matrix = vectorizer.fit_transform(corpus)
X = X_csr_matrix.todense()
vectorizer_vocabulary = vectorizer.vocabulary_
print('vectorizer fit transform return:\n{}'.format(X_csr_matrix))
print('return to dense:\n{}'.format(X))
print('vectorizer vocabulary:\n{}'.format(vectorizer_vocabulary))

corpus.append('I ate a sandwich')
X_csr_matrix = vectorizer.fit_transform(corpus)
X = X_csr_matrix.todense()
vectorizer_vocabulary = vectorizer.vocabulary_
print('vectorizer fit transform return:\n{}'.format(X_csr_matrix))
print('return to dense:\n{}'.format(X))
print('vectorizer vocabulary:\n{}'.format(vectorizer_vocabulary))

print('Distance between 1st and 2nd documents: {}'.format(euclidean_distances(X[0], X[1])))
print('Distance between 1st and 3rd documents: {}'.format(euclidean_distances(X[0], X[2])))
print('Distance between 2nd and 3rd documents: {}'.format(euclidean_distances(X[1], X[2])))
