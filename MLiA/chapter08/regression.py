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


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


def ridgeRegress(xMat, yMat, lam=0.2):
    n, m = np.shape(xMat)
    xTx = xMat.T * xMat
    denom = xTx + np.eye(m) * lam
    if np.linalg.det(denom) == 0.0:
        print('this matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr, numTestPts=30):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xMat = (xMat - xMat.mean(0)) / xMat.var(0)
    yMat = yMat - yMat.mean()
    m, n = xMat.shape
    wMat = np.zeros((numTestPts, n))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    else:
        return wMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xMat = (xMat - xMat.mean(0)) / xMat.var(0)
    yMat = yMat - yMat.mean()
    m, n = xMat.shape
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsMax = ws.copy()
    for i in range(numIt):
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, +1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        else:
            ws = wsMax.copy()
            returnMat[i, :] = ws.T
    else:
        return returnMat






