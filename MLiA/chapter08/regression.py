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
