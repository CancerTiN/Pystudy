import operator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from MLiA.chapter02 import kNN


def test1():
    print('#' * 64)
    group, labels = kNN.createDataSet()
    print('group: {}'.format(group))
    print('labels: {}'.format(labels))
    print('#' * 64)
    inX, k = [0, 0], 3
    print('inX: {}'.format(inX))
    print('k: {}'.format(k))
    predicted_label = kNN.classify0(inX, group, labels, k)
    print('predicted_label: {}'.format(predicted_label))
    print('#' * 64)


def test2():
    print('#' * 64)
    datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
    print('datingDataMat: {}'.format(datingDataMat))
    print('datingLabels: {}'.format(datingLabels))
    print('#' * 64)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
               15.0 * np.array(datingLabels),
               15.0 * np.array(datingLabels))
    plt.show()
    print('#' * 64)


def test3():
    print('#' * 64)
    datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
    print('datingDataMat: {}'.format(datingDataMat))
    print('datingLabels: {}'.format(datingLabels))
    normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
    print('#' * 64)
    print('normMat: {}'.format(normMat))
    print('ranges: {}'.format(ranges))
    print('minVals: {}'.format(minVals))
    print('#' * 64)


def test4():
    print('#' * 64)
    kNN.datingClassTest()
    print('#' * 64)


def test5():
    print('#' * 64)
    kNN.classifyPerson(ffMiles=10000, percentTats=0.5, iceCream=10)
    print('#' * 64)


def test6():
    print('#' * 64)
    testVector = kNN.img2vector('trainingDigits/0_13.txt')
    print('testVector: {}'.format(testVector))
    print('#' * 64)


def test7():
    print('#' * 64)
    kToErrorRates = list()
    for k in range(3, 34):
        errorRate = kNN.handwritingClassTest(k)
        kToErrorRates.append((k, errorRate))
        print('#' * 64)
    else:
        bestRank = sorted(kToErrorRates, key=operator.itemgetter(1))
        print('the groups including k and related error rate are:\n{}'.format(bestRank))
    print('#' * 64)


if __name__ == '__main__':
    test7()
