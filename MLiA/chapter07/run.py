import numpy as np

from MLiA.chapter07 import adaboost


def test0():
    print('#' * 64)
    dataMatrix, classLabels = adaboost.loadSimpData()
    m = len(classLabels)
    D = np.mat(np.ones((m, 1)) / m)
    print('dataMatrix: {}'.format(dataMatrix))
    print('classLabels: {}'.format(classLabels))
    print('D: {}'.format(D))
    print('#' * 64)
    bestStump, minError, bestClasEst = adaboost.buildStump(dataMatrix, classLabels, D)
    print('bestStump: {}'.format(bestStump))
    print('minError: {}'.format(minError))
    print('bestClasEst: {}'.format(bestClasEst))
    print('#' * 64)


def test1():
    print('#' * 64)
    dataMatrix, classLabels = adaboost.loadSimpData()
    print('dataMatrix: {}'.format(dataMatrix))
    print('classLabels: {}'.format(classLabels))
    print('#' * 64)
    classifierArray, aggClassEst = adaboost.adaBoostTrainDS(dataMatrix, classLabels)
    print('classifierArray: {}'.format(classifierArray))
    print('#' * 64)


def test2():
    print('#' * 64)
    dataMatrix, classLabels = adaboost.loadSimpData()
    print('dataMatrix: {}'.format(dataMatrix))
    print('classLabels: {}'.format(classLabels))
    print('#' * 64)
    classifierArray, aggClassEst = adaboost.adaBoostTrainDS(dataMatrix, classLabels)
    print('classifierArray: {}'.format(classifierArray))
    print('#' * 64)
    testData = [[5, 5], [1.1, 1.1], [1.1, 1], [1, 1], [0, 0]]
    print('testData: {}'.format(testData))
    testPred = adaboost.adaClassify(testData, classifierArray)
    print('testPred: {}'.format(testPred))
    print('#' * 64)


def test3():
    print('#' * 64)
    dataArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
    dataMat = np.mat(dataArr)
    classLabels = labelArr
    print('dataMat: {}'.format(dataMat))
    print('classLabels: {}'.format(classLabels))
    classifierArray, aggClassEst = adaboost.adaBoostTrainDS(dataMat, classLabels)
    print('#' * 64)
    testDataArr, testLabelArr = adaboost.loadDataSet('horseColicTest2.txt')
    testPredArr = adaboost.adaClassify(testDataArr, classifierArray)
    print('testLabelArr: {}'.format(testLabelArr))
    print('testPredArr: {}'.format(testPredArr))
    print('#' * 64)
    errArr = np.mat(np.ones((len(testLabelArr), 1)))
    errNum = errArr[np.mat(testLabelArr).T != testPredArr].sum()
    print('errNum: {}'.format(errNum))
    print('errRate: {}'.format(errNum / len(testLabelArr)))
    print('#' * 64)


def test4():
    print('#' * 64)
    dataArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
    dataMat = np.mat(dataArr)
    classLabels = labelArr
    classifierArray, aggClassEst = adaboost.adaBoostTrainDS(dataMat, classLabels)
    print('#' * 64)
    adaboost.plotROC(aggClassEst.T, classLabels)
    print('#' * 64)


if __name__ == '__main__':
    test4()
