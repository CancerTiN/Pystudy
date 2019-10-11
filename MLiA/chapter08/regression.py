import numpy as np
import pandas as pd


def loadDataSet(fileName):
    datFrame = pd.read_table(fileName, header=None)
    numFeat = datFrame.shape[1] - 1
    datMat = np.mat(datFrame.iloc[:, list(range(numFeat))].to_numpy())
    labelArr = datFrame.iloc[:, numFeat].to_numpy()
    return datMat, labelArr


def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print('this matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2 * k ** 2))
    else:
        xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print('this matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    else:
        return yHat
