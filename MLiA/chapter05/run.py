import numpy as np

from MLiA.chapter05 import logRegres


def test0():
    print('#' * 64)
    dataArr, labelMat = logRegres.loadDataSet()
    print('dataArr: {}'.format(dataArr))
    print('labelMat: {}'.format(labelMat))
    print('#' * 64)
    weights = logRegres.gradAscent(dataArr, labelMat)
    print('weights:\n{}'.format(weights))
    print('#' * 64)


def test1():
    print('#' * 64)
    dataArr, labelMat = logRegres.loadDataSet()
    weights = logRegres.gradAscent(dataArr, labelMat)
    logRegres.plotBestFit(weights.getA())
    print('#' * 64)


def test2():
    print('#' * 64)
    dataArr, labelMat = logRegres.loadDataSet()
    weights = logRegres.stocGradAscent0(np.array(dataArr), labelMat)
    logRegres.plotBestFit(weights)
    print('#' * 64)


def test3():
    print('#' * 64)
    dataArr, labelMat = logRegres.loadDataSet()
    weights = logRegres.stocGradAscent1(np.array(dataArr), labelMat)
    logRegres.plotBestFit(weights)
    print('#' * 64)


def test4():
    print('#' * 64)
    logRegres.multiTest()
    print('#' * 64)


if __name__ == '__main__':
    test4()
