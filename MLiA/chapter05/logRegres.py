import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def loadDataSet():
    dataMat, labelMat = list(), list()
    for line in open('testSet.txt'):
        lineArr = line.strip().split('\t')
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    else:
        return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        inX = dataMatrix * weights
        h = sigmoid(inX)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    else:
        boolStat = pd.DataFrame(labelMat == h.round())[0].value_counts()
        print('boolStat:\n{}'.format(boolStat))
        return weights


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    m, n = dataArr.shape
    xcord1, xcord2, ycord1, ycord2 = list(), list(), list(), list()
    for i in range(m):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x1 = np.arange(-3.0, 3.0, 0.1)
    x2 = (0 - weights[0] * 1 - weights[1] * x1) / weights[2]
    ax.plot(x1, x2)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    m, n = dataMatrix.shape
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(np.sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    else:
        return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = dataMatrix.shape
    weights = np.ones(n)
    for j in range(numIter):
        dataIndexes = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = random.randint(0, len(dataIndexes) - 1)
            h = sigmoid(np.sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            dataIndexes.pop(randIndex)
    else:
        return weights


def classifyVector(inX, weights):
    prob = sigmoid(np.dot(inX, weights))
    return 1.0 if prob > 0.5 else 0.0


def colicTest():
    trainingSet = list()
    trainingLabels = list()
    for line in open('horseColicTraining.txt'):
        currLine = line.strip().split('\t')
        lineArr = list(map(float, currLine[:-1]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    else:
        trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0.0
    numTestVec = 0.0
    for line in open('horseColicTest.txt'):
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = list(map(float, currLine[:-1]))
        classRet = int(classifyVector(np.array(lineArr), trainWeights))
        classRes = int(currLine[-1])
        if classRet != classRes:
            errorCount += 1.0
    else:
        errorRate = errorCount / numTestVec
        return errorRate


def multiTest():
    numTests, errorSum = 10, 0.0
    for k in range(numTests):
        errorRate = colicTest()
        print('errorRate: {}'.format(errorRate))
        errorSum += errorRate
    else:
        print('after {} iterations, the average error rate is {}'.format(numTests, errorSum / numTests))
