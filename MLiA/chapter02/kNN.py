import numpy as np
import operator
import os

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
    # print('minVals: {}'.format(minVals))
    maxVals = dataSet.max(0)
    # print('maxVals: {}'.format(maxVals))
    ranges = maxVals - minVals
    # print('ranges: {}'.format(ranges))
    m = dataSet.shape[0]
    # print('m: {}'.format(m))
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

def classifyPerson(ffMiles: float, percentTats:float, iceCream: float):
    print('frequent flier miles earned per years? ({})'.format(ffMiles))
    print('percentage of time spent playing video games? ({})'.format(percentTats))
    print('liters if ice cream consumde per year? ({})'.format(iceCream))
    resultList = ['not at all', 'in small doses', 'in large doses']
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    normArr = (inArr - minVals) / ranges
    classifierResult = classify0(normArr, normMat, datingLabels, 3)
    print('you will probably like this person: {}'.format(resultList[classifierResult - 1]))

def img2vector(filename):
    dataList = list()
    for line in open(filename):
        dataList.extend(map(int, list(line.strip())))
    else:
        return np.array(dataList)

def handwritingClassTest(k=3, trainingDatDir='trainingDigits', testDatDir='testDigits'):
    print('{} nearest neighbor will be considered'.format(k))
    hwLabels = list()
    trainingFileList = os.listdir(trainingDatDir)
    trainingMat = np.zeros((len(trainingFileList), 1024))
    getClassNum = lambda s: int(s.split('_')[0])
    for i, fileNameStr in enumerate(trainingFileList):
        hwLabels.append(getClassNum(fileNameStr))
        trainingMat[i,:] = img2vector(os.path.join(trainingDatDir, fileNameStr))
    else:
        pass
        # print('succeed in building data matrix from {} files'.format(i + 1))
    testFileList = os.listdir(testDatDir)
    errorCount = 0.0
    errorList = list()
    for i, fileNameStr in enumerate(testFileList):
        classNumInt = getClassNum(fileNameStr)
        vectorUnderTest = img2vector(os.path.join(testDatDir, fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, k)
        # print('the classifier came back with: {}, the real anwser is: {}'.format(classifierResult, classNumInt))
        if classifierResult != classNumInt:
            errorCount += 1.0
            errorList.append((fileNameStr, classifierResult))
    else:
        # print('the total number of errors is: {}'.format(errorCount))
        # print('the mis-predicted files with incorrect result are:\n{}'.format(errorList))
        print('the total error rate is: {}'.format(errorCount / (i + 1)))
        return errorCount / (i + 1)