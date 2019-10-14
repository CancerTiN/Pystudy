import numpy as np

from MLiA.chapter09 import regTrees


def test0():
    print('#' * 64)
    testMat = np.mat(np.eye(4))
    mat0, mat1 = regTrees.binSplitDataSet(testMat, 1, 0.5)
    print('mat0: {}'.format(mat0))
    print('mat1: {}'.format(mat1))
    print('#' * 64)


if __name__ == '__main__':
    test0()
