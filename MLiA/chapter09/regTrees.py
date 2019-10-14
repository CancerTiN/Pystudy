import numpy as np


def loadDataSet(fileName):
    return [map(float, line.strip().split('\t')) for line in open(fileName)]


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1
