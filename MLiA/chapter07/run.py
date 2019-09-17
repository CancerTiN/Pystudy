import numpy as np

from MLiA.chapter07 import adaboost


def test0():
    print('#' * 64)
    dataMatrix, classLabels = adaboost.loadSimpData()
    m = len(classLabels)
    D = np.mat(np.ones((m, 1)) / m)
    print('dataMatrix: {}'.format(dataMatrix))
    print('classLabels: {}'.format(classLabels))
    print('D: {}'.format(D))
    print('#' * 64)
    bestStump, minError, bestClasEst = adaboost.buildStump(dataMatrix, classLabels, D)
    print('bestStump: {}'.format(bestStump))
    print('minError: {}'.format(minError))
    print('bestClasEst: {}'.format(bestClasEst))
    print('#' * 64)


if __name__ == '__main__':
    test0()
