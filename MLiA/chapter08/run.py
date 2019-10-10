import numpy as np

from MLiA.chapter08 import regression


def test0():
    print('#' * 64)
    xArr, yArr = regression.loadDataSet('ex0.txt')
    print('xArr: {}'.format(xArr))
    print('yArr: {}'.format(yArr))
    print('#' * 64)
    ws = regression.standRegres(xArr, yArr)
    print('ws: {}'.format(ws))
    yHat = xArr * ws
    print('yHat: {}'.format(yHat))
    print('#' * 64)


if __name__ == '__main__':
    test0()
