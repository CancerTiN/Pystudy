# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

from sklearn.feature_extraction.text import CountVectorizer

corpus = ['The dog ate a sandwich, the wizard transfigured a sandwich, and I ate a sandwich']
vectorizer = CountVectorizer(stop_words='english')

X_csr_matrix = vectorizer.fit_transform(corpus)
X = X_csr_matrix.todense()
frequencies = X.getA1()
feature_names = vectorizer.get_feature_names()
token_indices = vectorizer.vocabulary_

print('vectorizer fit transform return:\n{}'.format(X_csr_matrix))
print('return to dense:\n{}'.format(X))
print('return to dense type:\n{}'.format(type(X)))
print('frequencies:\n{}'.format(frequencies))
print('feature names:\n{}'.format(feature_names))
print('token indices:\n{}'.format(token_indices))
for token, index in token_indices.items():
    print('The token "{}" appears {} times'.format(token, frequencies[index]))
