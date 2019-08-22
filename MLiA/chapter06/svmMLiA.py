import random
import numpy as np


def loadDataSet(filename):
    dataMat, labelMat = list(), list()
    for line in open(filename):
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    else:
        return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while j == i:
        j = random.randint(0, m)
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C=0.6, toler=0.001, maxIter=1):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < maxIter:
        alphasPairsChanged = 0
        for i in range(m):
            wMat = np.dot(np.multiply(alphas, labelMat).T, dataMatrix)
            print('wMat: {}'.format(wMat))
            print('wMat.shape: {}'.format(wMat.shape))
            xMat = dataMatrix[i, :].T
            print('xMat: {}'.format(xMat))
            print('xMat.shape: {}'.format(xMat.shape))
            wxMat = np.dot(wMat, xMat)
            print('wxMat: {}'.format(wxMat))
            print('wxMat.shape: {}'.format(wxMat.shape))
            fXi = wxMat - b
            print('fXi: {}'.format(fXi))
            print('fXi.shape: {}'.format(fXi.shape))
            Ei = fXi - labelMat[i]
            print('Ei: {}'.format(Ei))
            print('Ei.shape: {}'.format(Ei.shape))
        iter += 1
