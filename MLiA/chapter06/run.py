from MLiA.chapter06 import svmMLiA


def test0():
    print('#' * 64)
    dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
    print('dataArr: {}'.format(dataArr))
    print('labelArr: {}'.format(labelArr))
    print('#' * 64)


def test1():
    print('#' * 64)
    dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
    b, alphas = svmMLiA.smoSimple(dataArr, labelArr, C=0.6, toler=0.001, maxIter=20)
    print('b: {}'.format(b))
    print('alphas: {}'.format(alphas))
    print('#' * 64)
    k = 4
    print('display {} support vectors'.format(k))
    svmMLiA.plotSuportVecter(dataArr, labelArr, alphas, b, k=k)
    print('#' * 64)


def test2():
    print('#' * 64)
    dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
    b, alphas = svmMLiA.smoP(dataArr, labelArr, C=0.6, toler=0.001, maxIter=40)
    print('b: {}'.format(b))
    print('alphas: {}'.format(alphas))
    print('#' * 64)
    k = 4
    print('display {} support vectors'.format(k))
    svmMLiA.plotSuportVecter(dataArr, labelArr, alphas, float(b), k=k)
    print('#' * 64)


def test3():
    print('#' * 64)
    dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
    b, alphas = svmMLiA.smoP(dataArr, labelArr, C=0.6, toler=0.001, maxIter=40)
    print('b: {}'.format(b))
    print('alphas: {}'.format(alphas))
    print('#' * 64)
    k = 4
    print('display {} support vectors'.format(k))
    svmMLiA.plotSuportVecter(dataArr, labelArr, alphas, float(b), k=k)
    print('#' * 64)
    ws = svmMLiA.calcWs(alphas, dataArr, labelArr)
    print('ws: {}'.format(ws))
    correctCount, accuracy = svmMLiA.evalParams(dataArr, labelArr, ws, b)
    print('correct count: {}'.format(correctCount))
    print('accuracy: {}'.format(accuracy))
    print('#' * 64)

def test4():
    print('#' * 64)
    svmMLiA.testRbf()
    print('#' * 64)
    svmMLiA.testRbf(0.1)
    print('#' * 64)

def test5():
    print('#' * 64)
    trainingDataDir = 'D:/Workspace/Study/MLiA/chapter02/trainingDigits'
    testDataDir = 'D:/Workspace/Study/MLiA/chapter02/testDigits'
    svmMLiA.testDigits(trainingDataDir, testDataDir)
    print('#' * 64)

if __name__ == '__main__':
    test5()
