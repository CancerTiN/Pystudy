import numpy as np

from MLiA.chapter09 import regTrees


def test0():
    print('#' * 64)
    testMat = np.mat(np.eye(4))
    mat0, mat1 = regTrees.binSplitDataSet(testMat, 1, 0.5)
    print('mat0: {}'.format(mat0))
    print('mat1: {}'.format(mat1))
    print('#' * 64)


def test1():
    print('#' * 64)
    myDat = regTrees.loadDataSet('ex00.txt')
    myMat = np.mat(myDat)
    print('myMat: {}'.format(myMat))
    print('#' * 64)
    feat, val = regTrees.chooseBestSplit(myMat)
    print('feat: {}'.format(feat))
    print('val: {}'.format(val))
    print('#' * 64)


def test2():
    print('#' * 64)
    myDat = regTrees.loadDataSet('ex00.txt')
    myMat = np.mat(myDat)
    print('myMat: {}'.format(myMat))
    print('#' * 64)
    retTree = regTrees.createTree(myMat)
    print('retTree: {}'.format(retTree))
    print('#' * 64)


def test3():
    print('#' * 64)
    myDat = regTrees.loadDataSet('ex0.txt')
    myMat = np.mat(myDat)
    print('myMat: {}'.format(myMat))
    print('#' * 64)
    retTree = regTrees.createTree(myMat)
    print('retTree: {}'.format(retTree))
    print('#' * 64)


if __name__ == '__main__':
    test3()
