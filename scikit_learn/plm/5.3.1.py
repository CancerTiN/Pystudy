# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('winequality-red.csv', sep=';')
print(df.describe())

plt.scatter(df['alcohol'], df['quality'])
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol Against Quality')
plt.show()

X = df[df.columns[:-1]]
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_predictions = regressor.predict(X_test)
r_squared = regressor.score(X_test, y_test)
print('R-squared: {}'.format(r_squared))

scores = cross_val_score(LinearRegression(), X, y, cv=5)
scores_mean = scores.mean()
print('scores: {}'.format(scores))
print('scores mean: {}'.format(scores_mean))
