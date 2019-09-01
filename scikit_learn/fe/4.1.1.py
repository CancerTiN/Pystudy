# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

from sklearn.feature_extraction import DictVectorizer
onehot_encoder = DictVectorizer()

X = [{'city': 'New York'},
     {'city': 'San Francisco'},
     {'city': 'Chapel Hill'}]

ret = onehot_encoder.fit_transform(X)
print('fit transform return:\n{}'.format(ret))
print('fit transform return type: {}'.format(type(ret)))
ret_array = ret.toarray()
print('fit transform return array:\n{}'.format(ret_array))
print('fit transform return array type: {}'.format(type(ret_array)))
