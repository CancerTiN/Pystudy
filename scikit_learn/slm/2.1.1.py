# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[6], [8], [10], [14], [18]])
print('explanatory variable: {}'.format(X))
y = [7, 9, 13, 17.5, 18]
print('response variable: {}'.format(y))

plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X, y, 'k.')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()

model = LinearRegression()
model.fit(X, y)

test_pizza = np.array([[12]])
predicted_price = model.predict(test_pizza)[0]
print('A 12" pizza should cost: ${:.2f}'.format(predicted_price))

print('Residual sum of squares: {:.2f}'.format(np.mean((model.predict(X) - y) ** 2)))
