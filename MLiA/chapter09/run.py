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


def test4():
    print('#' * 64)
    myDat2 = regTrees.loadDataSet('ex2.txt')
    myMat2 = np.mat(myDat2)
    retTree = regTrees.createTree(myMat2, ops=(1, 4))
    print('retTree: {}'.format(retTree))
    print('ops: {}'.format((1, 4)))
    print('#' * 64)
    retTree = regTrees.createTree(myMat2, ops=(10000, 4))
    print('retTree: {}'.format(retTree))
    print('ops: {}'.format((10000, 4)))
    print('#' * 64)
    retTree = regTrees.createTree(myMat2, ops=(0, 1))
    print('retTree: {}'.format(retTree))
    print('ops: {}'.format((0, 1)))
    print('#' * 64)


def test5():
    print('#' * 64)
    myDat2 = regTrees.loadDataSet('ex2.txt')
    myMat2 = np.mat(myDat2)
    myTree = regTrees.createTree(myMat2, ops=(1, 4))
    print('myTree: {}'.format(myTree))
    print('ops: {}'.format((1, 4)))
    print('#' * 64)
    myDat2Test = regTrees.loadDataSet('ex2test.txt')
    myMat2Test = np.mat(myDat2Test)
    pruneTree = regTrees.prune(myTree, myMat2Test)
    print('pruneTree: {}'.format(pruneTree))


def test6():
    print('#' * 64)
    myMat2 = np.mat(regTrees.loadDataSet('exp2.txt'))
    print('myMat2: {}'.format(myMat2))
    print('#' * 64)
    myTree2 = regTrees.createTree(myMat2, regTrees.modelLeaf, regTrees.modelErr, (1, 10))
    print('myTree2: {}'.format(myTree2))
    print('#' * 64)


def test7():
    print('#' * 64)
    trainMat = np.mat(regTrees.loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = np.mat(regTrees.loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = regTrees.createTree(trainMat, ops=(1, 20))
    yHat = regTrees.createForeCase(myTree, testMat[:, 0])
    corrcoef = np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    print('corrcoef from regTree: {}'.format(corrcoef))
    print('#' * 64)
    trainMat = np.mat(regTrees.loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = np.mat(regTrees.loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = regTrees.createTree(trainMat, regTrees.modelLeaf, regTrees.modelErr, ops=(1, 20))
    yHat = regTrees.createForeCase(myTree, testMat[:, 0], regTrees.modelTreeEval)
    corrcoef = np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    print('corrcoef from modelTree: {}'.format(corrcoef))
    print('#' * 64)
    ws, X, Y = regTrees.linearSolve(trainMat)
    for i in range(np.shape(testMat)[0]):
        yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    else:
        corrcoef = np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    print('corrcoef from regression: {}'.format(corrcoef))
    print('#' * 64)


if __name__ == '__main__':
    test7()
