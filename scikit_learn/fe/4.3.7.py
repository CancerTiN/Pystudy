# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

from sklearn.feature_extraction.text import HashingVectorizer

corpus = ['the', 'ate', 'bacon', 'cat']

vectorizer = HashingVectorizer(n_features=6)
csr_matrix = vectorizer.fit_transform(corpus)
hashing_weights = csr_matrix.todense()

print(hashing_weights)
