import numpy as np


def loadSimpData():
    dataMatrix = np.mat([[1.0, 2.1],
                         [2.0, 1.1],
                         [1.3, 1.0],
                         [1.0, 1.0],
                         [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMatrix, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    m, n = dataMatrix.shape
    retArray = np.ones((m, 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] >= threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = dataMatrix.shape
    numSteps = 10
    bestStump = dict()
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:, 1].min()
        rangeMax = dataMatrix[:, 1].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, numSteps + 1):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + j * stepSize
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = float(D.T * errArr)
                print('dim: {}, thresh: {}, ineqal: {}, weighted error: {}'.format(
                    i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    else:
        return bestStump, minError, bestClasEst
