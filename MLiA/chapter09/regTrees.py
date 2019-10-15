import numpy as np


def loadDataSet(fileName):
    return [list(map(float, line.strip().split('\t'))) for line in open(fileName)]


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].getA1()):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestValue = splitVal
    if S - bestS < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree = {
        'spInd': feat,
        'spVal': val,
        'left': createTree(lSet, leafType, errType, ops),
        'right': createTree(rSet, leafType, errType, ops),
    }
    return retTree
