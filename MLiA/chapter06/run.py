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


if __name__ == '__main__':
    test1()
