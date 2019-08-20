from MLiA.chapter06 import svmMLiA


def test0():
    print('#' * 64)
    dataArr, labelArr = svmMLiA.loadDataSet('testSet.txt')
    print('dataArr: {}'.format(dataArr))
    print('labelArr: {}'.format(labelArr))
    print('#' * 64)


if __name__ == '__main__':
    test0()
