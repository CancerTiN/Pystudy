import matplotlib.pyplot as plt
import numpy as np

from MLiA.chapter08 import regression


def test0():
    print('#' * 64)
    xArr, yArr = regression.loadDataSet('ex0.txt')
    print('xArr: {}'.format(xArr))
    print('yArr: {}'.format(yArr))
    print('#' * 64)
    ws = regression.standRegres(xArr, yArr)
    print('ws: {}'.format(ws))
    yHat = xArr * ws
    print('yHat: {}'.format(yHat))
    print('#' * 64)


def test1():
    print('#' * 64)
    xArr, yArr = regression.loadDataSet('ex0.txt')
    ws = regression.standRegres(xArr, yArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()
    print('#' * 64)
    corrArr = np.corrcoef(yHat.T, yMat)
    print('corrArr: {}'.format(corrArr))
    print('#' * 64)


def test2():
    print('#' * 64)
    xArr, yArr = regression.loadDataSet('ex0.txt')
    print('x: {}, y: {}'.format(xArr[0], yArr[0]))
    print('#' * 64)
    yHat0 = regression.lwlr(xArr[0], xArr, yArr, 1.0)
    print('x: {}, y: {}, k: {}'.format(xArr[0], yHat0, 1.0))
    yHat1 = regression.lwlr(xArr[0], xArr, yArr, 0.001)
    print('x: {}, y: {}, k: {}'.format(xArr[0], yHat1, 0.001))
    print('#' * 64)
    for k in [1.0, 0.01, 0.003]:
        yHat = regression.lwlrTest(xArr, xArr, yArr, k)
        xMat = np.mat(xArr)
        yMat = np.mat(yArr)
        srtInd = xMat[:, 1].argsort(0)
        xSort = xMat[:, 1].getA1()[srtInd]
        ySort = yHat[srtInd]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('k = {}'.format(k))
        ax.plot(xSort, ySort)
        ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0], s=2, c='red')
        plt.show()
    print('#' * 64)


def test3():
    print('#' * 64)
    abX, abY = regression.loadDataSet('abalone.txt')
    print('#' * 64)
    a = 100
    b = 199
    print('test data set from {} to {}'.format(a, b))
    for k in (0.1, 1, 10):
        yHat = regression.lwlrTest(abX[a:b], abX, abY, k)
        rssError = regression.rssError(abY[a:b], yHat.T)
        print('get rssError ({}) from k ({})'.format(rssError, k))
    print('#' * 64)
    ws = regression.standRegres(abX, abY)
    yHat = np.mat(abX) * ws
    rssError = regression.rssError(abY[a: b], yHat[a: b].T.A)
    print('get rssError ({}) from stand regression'.format(rssError))
    print('#' * 64)


def test4():
    print('#' * 64)
    abX, abY = regression.loadDataSet('abalone.txt')
    ridgeWeights = regression.ridgeTest(abX, abY)
    print('ridgeWeights: {}'.format(ridgeWeights))
    print('ridgeWeights shape: {}'.format(ridgeWeights.shape))
    print('#' * 64)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridgeWeights)
    # ax.plot(np.r_[ridgeWeights, [range(8)]])
    # plt.show()
    print('#' * 64)


def test5():
    print('#' * 64)
    abX, abY = regression.loadDataSet('abalone.txt')
    print('#' * 64)
    for eps, numIt in ((0.01, 200), (0.001, 5000)):
        returnMat = regression.stageWise(abX, abY, eps, numIt)
        print('returnMat ({}) from eps ({}) numIt ({})'.format(returnMat, eps, numIt))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(returnMat)
        plt.show()
        print('#' * 64)
    else:
        xMat = np.mat(abX)
        yMat = np.mat(abY).T
        xMat = (xMat - xMat.mean(0)) / xMat.var(0)
        yMat = yMat - yMat.mean()
        weights = regression.standRegres(xMat, yMat.T)
        print('weights ({}) from stand regression'.format(weights.T))
    print('#' * 64)


if __name__ == '__main__':
    test5()
