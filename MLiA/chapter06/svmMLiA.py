import random


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




