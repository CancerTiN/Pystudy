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


if __name__ == '__main__':
    test2()
