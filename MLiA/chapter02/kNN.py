import numpy as np
import operator

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # print('dataSetSize: {}'.format(dataSetSize))
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # print('diffMat: {}'.format(diffMat))
    sqDiffMat = diffMat ** 2
    # print('sqDiffMat: {}'.format(sqDiffMat))
    sqDistances = sqDiffMat.sum(axis=1)
    # print('sqDistances: {}'.format(sqDistances))
    distances = sqDistances ** 0.5
    # print('distances: {}'.format(distances))
    sortedDistIndicies = distances.argsort()
    # print('sortedDistIndicies: {}'.format(sortedDistIndicies))
    classCount = dict()
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # else:
    #     print('classCount: {}'.format(classCount))
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print('sortedClassCount: {}'.format(sortedClassCount))
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

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    print('minVals: {}'.format(minVals))
    maxVals = dataSet.max(0)
    print('maxVals: {}'.format(maxVals))
    ranges = maxVals - minVals
    print('ranges: {}'.format(ranges))
    m = dataSet.shape[0]
    print('m: {}'.format(m))
    normDataSet = (dataSet - np.tile(minVals, (m, 1))) / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],3)
        print('the classifier came back with: {}, the real answer is: {}'.format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    else:
        print('the total error rate is: {}'.format(errorCount / numTestVecs))
