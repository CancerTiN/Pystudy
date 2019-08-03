from MLIA.chapter02 import kNN


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
