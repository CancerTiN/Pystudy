from MLiA.chapter03 import trees

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

if __name__ == '__main__':
    test2()
