from MLiA.chapter02 import kNN
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def test1():
    print('#' * 64)
    group, labels = kNN.createDataSet()
    print('group: {}'.format(group))
    print('labels: {}'.format(labels))
    print('#' * 64)
    inX, k = [0, 0], 3
    print('inX: {}'.format(inX))
    print('k: {}'.format(k))
    predicted_label = kNN.classify0(inX, group, labels, k)
    print('predicted_label: {}'.format(predicted_label))
    print('#' * 64)

def test2():
    print('#' * 64)
    datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
    print('datingDataMat: {}'.format(datingDataMat))
    print('datingLabels: {}'.format(datingLabels))
    print('#' * 64)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2],
               15.0 * np.array(datingLabels),
               15.0 * np.array(datingLabels))
    plt.show()
    print('#' * 64)

if __name__ == '__main__':
    test2()
