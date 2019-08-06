import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    print('dataSetSize: {}'.format(dataSetSize))
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    print('diffMat: {}'.format(diffMat))
    sqDiffMat = diffMat ** 2
    print('sqDiffMat: {}'.format(sqDiffMat))
    sqDistances = sqDiffMat.sum(axis=1)
    print('sqDistances: {}'.format(sqDistances))
    distances = sqDistances ** 0.5
    print('distances: {}'.format(distances))
    sortedDistIndicies = distances.argsort()
    print('sortedDistIndicies: {}'.format(sortedDistIndicies))
    classCount = dict()
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    else:
        print('classCount: {}'.format(classCount))
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print('sortedClassCount: {}'.format(sortedClassCount))
    return sortedClassCount[0][0]

def file2matrix(filename):
    arrayOfLines = open(filename).readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = list()
    for index, line in enumerate(arrayOfLines):
        listFromLine = line.strip().split('\t')
        returnMat[index, :] = listFromLine[:3]
        classLabelVector.append(int(listFromLine[-1]))
    else:
        return returnMat, classLabelVector

