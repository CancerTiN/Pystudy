import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        j = random.randint(0, m - 1)
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C=0.6, toler=0.001, maxIter=60, bChangeRate=0.25):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.random.rand(m, 1))
    alphas[alphas > C] = 0
    # alphas = np.mat(np.zeros((m, 1)))
    count = 0
    iter = 0

    x1cord0, x2cord0, x1cord1, x2cord1 = list(), list(), list(), list()
    for row, label in zip(dataMatrix, labelMat):
        x1Pt, x2Pt = row.getA1()
        if label == -1:
            x1cord0.append(x1Pt)
            x2cord0.append(x2Pt)
        elif label == 1:
            x1cord1.append(x1Pt)
            x2cord1.append(x2Pt)
    plt.figure()

    while iter < maxIter:
        maxSum = 0
        alphasPairsChanged = 0
        for i in range(m):
            lMat = np.multiply(alphas, labelMat).T
            wMat = np.dot(lMat, dataMatrix)
            xiMat = dataMatrix[i, :]
            wXi = np.dot(wMat, xiMat.T)
            fXi = wXi - b
            Ei = fXi - labelMat[i]
            # cond1为Xi与间隔平面的负距离的情况，
            # 可能是分类正确但Xi处在分隔超平面和间隔平面之间，也可能是分类错误，
            # 如果对应的alpha还没有超过定义域的右侧，
            # 需要增大该alpha以使Xi远离间隔平面。
            cond1 = labelMat[i] * Ei < -toler and alphas[i] < C
            # cond2为Xi与间隔平面的正距离的情况，
            # 这是分类正确的情况，
            # 如果对应的alpha不为零则会使Xi参与到参数w的线下构成中，
            # 需要将该alpha置零以忽略Xi对参数的影响。
            cond2 = labelMat[i] * Ei > +toler and alphas[i] > 0
            if cond1 or cond2:
                j = selectJrand(i, m)
                xjMat = dataMatrix[j, :]
                wXj = np.dot(wMat, xjMat.T)
                fXj = wXj - b
                Ej = fXj - labelMat[j]
                aiOld = alphas[i].copy()
                ajOld = alphas[j].copy()
                if labelMat[i] != labelMat[j]:
                    L = max(0, ajOld - aiOld)
                    H = min(C, C + ajOld - aiOld)
                else:
                    L = max(0, ajOld + aiOld - C)
                    H = min(C, ajOld + aiOld)
                if L == H:
                    continue
                coorDiff = xiMat - xjMat
                euclDist = np.dot(coorDiff, coorDiff.T)
                if euclDist == 0:
                    continue
                ajNew = ajOld + (labelMat[j] * Ei - labelMat[j] * Ej) / euclDist
                ajNew = clipAlpha(ajNew, H, L)
                aiNew = aiOld + labelMat[i] * labelMat[j] * (ajOld - ajNew)
                if abs(ajNew - ajOld) < 1e-5:
                    continue
                alphas[j] = ajNew
                alphas[i] = aiNew
                bi = labelMat[i] - np.dot(wMat, xiMat.T)
                bj = labelMat[j] - np.dot(wMat, xjMat.T)
                if 0 < alphas[i] < C and 0 < alphas[j] < C:
                    bTar = np.mean((bi, bj))
                elif 0 < alphas[i] < C:
                    bTar = float(bi)
                elif 0 < alphas[j] < C:
                    bTar = float(bj)
                else:
                    bTar = np.mean((bi, bj))
                b = b + (bTar - b) * bChangeRate
                alphasPairsChanged += 1
        if alphasPairsChanged == 0:
            iter += 1
        else:
            iter = 0

        w1, w2 = np.dot(np.multiply(alphas, labelMat).T, dataMatrix).getA1()
        x1tarr = np.arange(-4, 14, 0.1)
        x2tarr = (-w1 * x1tarr - b) / w2

        plt.clf()
        plt.title('{}-{}'.format(count, iter))
        plt.axis([-4, 14, -10, 8])
        plt.scatter(x1cord0, x2cord0, marker='s', s=90, c='blue')
        plt.scatter(x1cord1, x2cord1, marker='o', s=60, c='red')
        plt.plot(x1tarr, x2tarr, c='black')
        plt.pause(0.01)

        count += 1

    plt.waitforbuttonpress()
    plt.close()
    return b, alphas


def plotSuportVecter(dataMatIn, classLabels, alphas, b, k=3):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    w1, w2 = np.dot(np.multiply(alphas, labelMat).T, dataMatrix).getA1()
    x1cord0, x2cord0, x1cord1, x2cord1 = list(), list(), list(), list()
    cord2distList = list()
    for row, label in zip(dataMatrix, labelMat):
        x1Pt, x2Pt = row.getA1()
        if label == -1:
            x1cord0.append(x1Pt)
            x2cord0.append(x2Pt)
        elif label == 1:
            x1cord1.append(x1Pt)
            x2cord1.append(x2Pt)
        dist = label * (w1 * x1Pt + w2 * x2Pt + b)
        cord2distList.append([(x1Pt, x2Pt), dist])
    else:
        cord2distDF = pd.DataFrame(cord2distList)
        svCordList = cord2distDF.sort_values(1).head(k)[0].to_list()
    x1tarr = np.arange(-4, 14, 0.1)
    x2tarr = (-w1 * x1tarr - b) / w2

    plt.figure()
    plt.axis([-4, 14, -10, 8])
    plt.scatter(x1cord0, x2cord0, marker='s', s=90, c='blue')
    plt.scatter(x1cord1, x2cord1, marker='o', s=60, c='red')
    plt.plot(x1tarr, x2tarr, c='black')
    for svCord in svCordList:
        circle = plt.Circle(svCord, 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
        plt.gca().add_patch(circle)
    else:
        plt.waitforbuttonpress()


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = dataMatIn.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))

    def pickLH(self, i: int, j: int):
        if self.labelMat[i] != self.labelMat[j]:
            L = max(0, self.alphas[j] - self.alphas[i])
            H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
        else:
            L = max(0, self.alphas[j] + self.alphas[i] - self.C)
            H = min(self.C, self.alphas[j] + self.alphas[i])
        return L, H


def calcEk(oS, k: int):
    if isinstance(oS, optStruct):
        fXk = np.dot(np.multiply(oS.alphas, oS.labelMat).T, oS.X) * oS.X[k, :].T + oS.b
    elif isinstance(oS, optStructK):
        fXk = np.dot(np.multiply(oS.alphas, oS.labelMat).T, oS.K[:,k]) + oS.b
    Ek = fXk - oS.labelMat[k]
    return float(Ek)


def selectJ(i: int, oS: optStruct, Ei: float):
    maxK, maxDeltaE, Ej = -1, 0, 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k != i:
                Ek = calcEk(oS, k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK, maxDeltaE, Ej = k, deltaE, Ek
        else:
            j = maxK
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS: optStruct, k: int):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i: int, oS: optStruct):
    Ei = calcEk(oS, i)
    cond1 = oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C
    cond2 = oS.labelMat[i] * Ei > +oS.tol and oS.alphas[i] > 0
    if cond1 or cond2:
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        L, H = oS.pickLH(i, j)
        if L == H:
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 1e-5:
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    dataMatIn = np.mat(dataMatIn)
    classLabels = np.mat(classLabels).transpose()
    oS = optStruct(dataMatIn, classLabels, C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    x1cord0, x2cord0, x1cord1, x2cord1 = list(), list(), list(), list()
    for row, label in zip(dataMatIn, classLabels):
        x1Pt, x2Pt = row.getA1()
        if label == -1:
            x1cord0.append(x1Pt)
            x2cord0.append(x2Pt)
        elif label == 1:
            x1cord1.append(x1Pt)
            x2cord1.append(x2Pt)
    plt.figure()

    while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print('full-set, iter: {} i: {}, pairs changed: {}'.format(iter, i, alphaPairsChanged))

                w1, w2 = np.dot(np.multiply(oS.alphas, oS.labelMat).T, oS.X).getA1()
                x1tarr = np.arange(-4, 14, 0.1)
                x2tarr = (-w1 * x1tarr - float(oS.b)) / w2

                plt.clf()
                plt.title('{}-{}'.format(iter, i))
                plt.axis([-4, 14, -10, 8])
                plt.scatter(x1cord0, x2cord0, marker='s', s=90, c='blue')
                plt.scatter(x1cord1, x2cord1, marker='o', s=60, c='red')
                plt.plot(x1tarr, x2tarr, c='black')
                plt.pause(0.01)

            else:
                iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print('non-bound, iter: {} i: {}, pairs changed: {}'.format(iter, i, alphaPairsChanged))

                w1, w2 = np.dot(np.multiply(oS.alphas, oS.labelMat).T, oS.X).getA1()
                x1tarr = np.arange(-4, 14, 0.1)
                x2tarr = (-w1 * x1tarr - float(oS.b)) / w2

                plt.clf()
                plt.title('{}-{}'.format(iter, i))
                plt.axis([-4, 14, -10, 8])
                plt.scatter(x1cord0, x2cord0, marker='s', s=90, c='blue')
                plt.scatter(x1cord1, x2cord1, marker='o', s=60, c='red')
                plt.plot(x1tarr, x2tarr, c='black')
                plt.pause(0.01)

            else:
                iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print('iteration number: {}'.format(iter))

    plt.waitforbuttonpress()
    plt.close()

    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    labelMat = np.mat(classLabels).transpose()
    return np.dot(np.multiply(alphas, labelMat).T, dataArr).T


def evalParams(dataArr, labelArr, ws, b):
    datMat = np.mat(dataArr)
    m, n = datMat.shape
    b = float(b)
    correctCount = 0.0
    for i in range(m):
        y_pred = datMat[i] * ws + b
        if np.sign(y_pred) == labelArr[i]:
            correctCount += 1
    else:
        return correctCount, correctCount / m

def kernelTrans(X: np.matrix, A: np.matrix, kTup: tuple):
    m, n = X.shape
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X *A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        else:
            K = np.exp(K/(-1*kTup[1]**2))
    return K

class optStructK:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = dataMatIn.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = self.transform(kTup)

    def transform(self, kTup):
        K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
        else:
            return K

    def pickLH(self, i: int, j: int):
        if self.labelMat[i] != self.labelMat[j]:
            L = max(0, self.alphas[j] - self.alphas[i])
            H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
        else:
            L = max(0, self.alphas[j] + self.alphas[i] - self.C)
            H = min(self.C, self.alphas[j] + self.alphas[i])
        return L, H

def innerK(i: int, oS: optStructK):
    Ei = calcEk(oS, i)
    cond1 = oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C
    cond2 = oS.labelMat[i] * Ei > +oS.tol and oS.alphas[i] > 0
    if cond1 or cond2:
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        L, H = oS.pickLH(i, j)
        if L == H:
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 1e-5:
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoPK(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    dataMatIn = np.mat(dataMatIn)
    classLabels = np.mat(classLabels).transpose()
    oS = optStructK(dataMatIn, classLabels, C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print('full-set, iter: {} i: {}, pairs changed: {}'.format(iter, i, alphaPairsChanged))
            else:
                iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print('non-bound, iter: {} i: {}, pairs changed: {}'.format(iter, i, alphaPairsChanged))
            else:
                iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print('iteration number: {}'.format(iter))
    return oS.b, oS.alphas

def testRbf(k1=1.3):
    kTup = ('rbf', k1)
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoPK(dataArr, labelArr, 200, 1e-4, 10000, kTup)
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print('there are {} support vectors'.format(sVs.shape[0]))
    m, n = datMat.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:], kTup)
        predict = float(kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b)
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    else:
        print('the training error rate is {}'.format(errorCount / m))
    dataArr2, labelArr2 = loadDataSet('testSetRBF2.txt')
    datMat2 = np.mat(dataArr2)
    labelMat2 = np.mat(labelArr2).transpose()
    m2, n2 = datMat2.shape
    errorCount2 = 0
    for j in range(m2):
        kernelEval2 = kernelTrans(sVs, datMat2[j, :], kTup)
        predict2 = float(kernelEval2.T * np.multiply(labelSV, alphas[svInd]) + b)
        if np.sign(predict2) != np.sign(labelArr[j]):
            errorCount2 += 1
    else:
        print('the test error rate is {}'.format(errorCount / m2))

def img2vector(filename):
    dataList = list()
    for line in open(filename):
        dataList.extend(map(int, list(line.strip())))
    else:
        return np.array(dataList)

def loadImages(dirname):
    hwLabels = list()
    trainningFileList = os.listdir(dirname)
    m = len(trainningFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        filename = trainningFileList[i]
        classNum = int(filename.split('_')[0])
        if classNum == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:] = img2vector(os.path.join(dirname, filename))
    else:
        return trainingMat, hwLabels

def testDigits(trainingDataDir, testDataDir, kTup=('rbf', 10)):
    dataArr, labelArr = loadImages(trainingDataDir)
    print('succeed in loading images from {}'.format(trainingDataDir))
    b, alphas = smoPK(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print('there are {} support vectors'.format(sVs.shape[0]))
    m, n = datMat.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:], kTup)
        predict = float(kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b)
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    else:
        print('the training error rate is {}'.format(errorCount / m))
    dataArr2, labelArr2 = loadImages(testDataDir)
    print('succeed in loading images from {}'.format(testDataDir))
    datMat2 = np.mat(dataArr2)
    labelMat2 = np.mat(labelArr2).transpose()
    m2, n2 = datMat2.shape
    errorCount2 = 0
    for j in range(m2):
        kernelEval2 = kernelTrans(sVs, datMat2[j,:], kTup)
        predict2 = float(kernelEval2.T * np.multiply(labelSV, alphas[svInd]) + b)
        if np.sign(predict2) != np.sign(labelArr2[j]):
            errorCount2 += 1
    else:
        print('the test error rate is {}'.format(errorCount2 / m2))
