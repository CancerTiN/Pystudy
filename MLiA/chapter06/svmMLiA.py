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
        j = random.randint(0, m - 1)
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
            lMat = np.multiply(alphas, labelMat).T
            wMat = np.dot(lMat, dataMatrix)
            xiMat = dataMatrix[i,:]
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
                xjMat = dataMatrix[j,:]
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
                ajNew = ajOld + (labelMat[i] * Ei - labelMat[j] * Ej) / euclDist
                ajNew = clipAlpha(ajNew, H, L)
                if abs(ajNew - ajOld) < 1e-5:
                    alphas[j] = ajNew
                    continue
                aiNew = aiOld + labelMat[i] * labelMat[j] * (ajOld - ajNew)
                alphas[i] = aiNew
                wMat = np.dot(np.multiply(alphas, labelMat).T, dataMatrix)
                bi = labelMat[i] - np.dot(wMat, xiMat.T)
                bj = labelMat[j] - np.dot(wMat, xjMat.T)
                if 0 < alphas[i] < C and 0 < alphas[j] < C:
                    b = np.mean(bi, bj)
                elif 0 < alphas[i] < C:
                    b = bi
                elif 0 < alphas[j] < C:
                    b = bj
                else:
                    b = np.mean(bi, bj)
                alphasPairsChanged += 1
        if alphasPairsChanged == 0:
            iter += 1
        else:
            iter = 0
    return b, alphas
