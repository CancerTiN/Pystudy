from MLiA.chapter03 import trees
from MLiA.chapter03 import treesPlotter

def test0():
    print('#' * 64)
    myDat, labels = trees.createDataSet()
    print('myDat: {}'.format(myDat))
    print('labels: {}'.format(labels))
    print('#' * 64)
    shannonEnt = trees.calcShannonEnt(myDat)
    print('shannonEnt: {}'.format(shannonEnt))
    print('#' * 64)

def test1():
    print('#' * 64)
    myDat, labels = trees.createDataSet()
    trees.chooseBestFeatureToSplit(myDat)
    print('#' * 64)

def test2():
    print('#' * 64)
    myDat, labels = trees.createDataSet()
    print('myDat: {}'.format(myDat))
    print('labels: {}'.format(labels))
    print('#' * 64)
    myTree = trees.createTree(myDat, labels)
    print('#' * 64)
    print('myTree: {}'.format(myTree))
    print('#' * 64)

def test3():
    print('#' * 64)
    treesPlotter.createPlot()
    print('#' * 64)

def test4():
    print('#' * 64)
    myTree = treesPlotter.retrieveTree(0)
    print('myTree: {}'.format(myTree))
    numLeafs = treesPlotter.getNumLeafs(myTree)
    print('numLeafs: {}'.format(numLeafs))
    treeDepth = treesPlotter.getTreeDepth(myTree)
    print('treeDepth: {}'.format(treeDepth))
    print('#' * 64)

def test5():
    print('#' * 64)
    myTree = treesPlotter.retrieveTree(1)
    print('myTree: {}'.format(myTree))
    print('#' * 64)
    treesPlotter.createPlot(myTree)
    print('#' * 64)

def test6():
    print('#' * 64)
    myDat, labels = trees.createDataSet()
    myTree = treesPlotter.retrieveTree(0)
    ret1 = trees.classify(myTree, labels, [1, 0])
    print('decision by {} on {} turns out {}'.format([1, 0], labels, ret1))
    ret2 = trees.classify(myTree, labels, [1, 1])
    print('decision by {} on {} turns out {}'.format([1, 1], labels, ret2))
    print('#' * 64)

def test7():
    print('#' * 64)
    myTree = treesPlotter.retrieveTree(0)
    trees.storeTree(myTree, 'classifierStorage.txt')
    reTree = trees.grabTree('classifierStorage.txt')
    print('reTree: {}'.format(reTree))
    print('#' * 64)

def test8():
    print('#' * 64)
    dataSet = [line.strip().split('\t') for line in open('lenses.txt')]
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = trees.createTree(dataSet, labels)
    print('lensesTree: {}'.format(lensesTree))
    treesPlotter.createPlot(lensesTree)
    print('#' * 64)

if __name__ == '__main__':
    test0()
