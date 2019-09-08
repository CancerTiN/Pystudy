# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ['The dog ate a sandwich and I ate a sandwich',
          'The wizard transfigured a sandwich']

vectorizer = TfidfVectorizer(stop_words='english')
csr_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()
tfidf_weights = csr_matrix.todense()

print(feature_names)
print(tfidf_weights)
