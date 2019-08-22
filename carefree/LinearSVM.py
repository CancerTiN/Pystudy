import matplotlib.pyplot as plt
import numpy as np


class LinearSVM:
    def __init__(self):
        self._w = self._b = None

    def fit(self, x, y, c=1, lr=0.01, epoch=1000, plot=False):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        self._w = np.zeros(x.shape[1])
        self._b = 0.
        if plot:
            axis, labels = np.array(x).T, np.array(y)
            decision_function = lambda xx: self.predict(xx)
            nx, ny, padding = 400, 400, 0.2
            x_min, x_max = np.min(axis[0]), np.max(axis[0])
            y_min, y_max = np.min(axis[1]), np.max(axis[1])
            x_padding = max(abs(x_min), abs(x_max)) * padding
            y_padding = max(abs(y_min), abs(y_max)) * padding
            x_min -= x_padding
            x_max += x_padding
            y_min -= y_padding
            y_max += y_padding

            def get_base(nx, ny):
                xf = np.linspace(x_min, x_max, nx)
                yf = np.linspace(y_min, y_max, ny)
                n_xf, n_yf = np.meshgrid(xf, yf)
                return xf, yf, np.c_[n_xf.ravel(), n_yf.ravel()]

            plt.figure()

        for _ in range(epoch):
            self._w *= 1 - lr
            err = 1 - y * self.predict(x, True)
            idx = np.argmax(err)
            # 注意即使所有 x, y 都满足 w·x + b >= 1
            # 由于损失里面有一个 w 的模长平方
            # 所以仍然不能终止训练，只能截断当前的梯度下降
            if err[idx] <= 0:
                continue
            delta = lr * c * y[idx]
            self._w += delta * x[idx]
            self._b += delta
            if plot:
                plt.clf()
                plt.title(str(_))
                plt.scatter(axis[0], axis[1], c=labels)
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                xf, yf, base_matrix = get_base(nx, ny)
                z = decision_function(base_matrix).reshape((nx, ny))
                plt.contour(xf, yf, z, levels=[0])
                plt.pause(0.01)
        else:
            plt.waitforbuttonpress()

    def predict(self, x, raw=False):
        x = np.asarray(x, np.float32)
        y_pred = x.dot(self._w) + self._b
        if raw:
            return y_pred
        return np.sign(y_pred).astype(np.float32)

class LinearSVM2(LinearSVM):
    # 用参数 batch_size 表示 Top n 中的 n
    def fit(self, x, y, c=1, lr=0.01, batch_size=128, epoch=1000, plot=False):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        # 如果 batch_size 设得比样本总数还多、则将其改为样本总数
        batch_size = min(batch_size, len(y))
        self._w = np.zeros(x.shape[1])
        self._b = 0.
        if plot:
            axis, labels = np.array(x).T, np.array(y)
            decision_function = lambda xx: self.predict(xx)
            nx, ny, padding = 400, 400, 0.2
            x_min, x_max = np.min(axis[0]), np.max(axis[0])
            y_min, y_max = np.min(axis[1]), np.max(axis[1])
            x_padding = max(abs(x_min), abs(x_max)) * padding
            y_padding = max(abs(y_min), abs(y_max)) * padding
            x_min -= x_padding
            x_max += x_padding
            y_min -= y_padding
            y_max += y_padding

            def get_base(nx, ny):
                xf = np.linspace(x_min, x_max, nx)
                yf = np.linspace(y_min, y_max, ny)
                n_xf, n_yf = np.meshgrid(xf, yf)
                return xf, yf, np.c_[n_xf.ravel(), n_yf.ravel()]

            plt.figure()
        for _ in range(epoch):
            self._w *= 1 - lr
            err = 1 - y * self.predict(x, True)
            # 利用 argsort 函数直接取出 Top n
            # 注意 argsort 的结果是从小到大的，所以要用 [::-1] 把结果翻转一下
            batch = np.argsort(err)[-batch_size:][::-1]
            err = err[batch]
            if err[0] <= 0:
                continue
            # 注意这里我们只能利用误分类的样本做梯度下降
            # 因为被正确分类的样本处、这一部分的梯度为 0
            mask = err > 0
            batch = batch[mask]
            # 取各梯度平均并做一步梯度下降
            delta = lr * c * y[batch]
            self._w += np.mean(delta[..., None] * x[batch], axis=0)
            self._b += np.mean(delta)

            if plot:
                plt.clf()
                plt.title(str(_))
                plt.scatter(axis[0], axis[1], c=labels)
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                xf, yf, base_matrix = get_base(nx, ny)
                z = decision_function(base_matrix).reshape((nx, ny))
                plt.contour(xf, yf, z, levels=[0])
                plt.pause(0.01)

class LinearSVM3(LinearSVM):
    def fit(self, x, y, c=1, lr=0.01, batch_size=128, epoch=10000, plot=False):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        batch_size = min(batch_size, len(y))
        self._w = np.zeros(x.shape[1])
        self._b = 0.
        if plot:
            axis, labels = np.array(x).T, np.array(y)
            decision_function = lambda xx: self.predict(xx)
            nx, ny, padding = 400, 400, 0.2
            x_min, x_max = np.min(axis[0]), np.max(axis[0])
            y_min, y_max = np.min(axis[1]), np.max(axis[1])
            x_padding = max(abs(x_min), abs(x_max)) * padding
            y_padding = max(abs(y_min), abs(y_max)) * padding
            x_min -= x_padding
            x_max += x_padding
            y_min -= y_padding
            y_max += y_padding

            def get_base(nx, ny):
                xf = np.linspace(x_min, x_max, nx)
                yf = np.linspace(y_min, y_max, ny)
                n_xf, n_yf = np.meshgrid(xf, yf)
                return xf, yf, np.c_[n_xf.ravel(), n_yf.ravel()]

            plt.figure()

        for _ in range(epoch):
            self._w *= 1 - lr
            # 随机选取 batch_size 个样本
            batch = np.random.choice(len(x), batch_size)
            x_batch, y_batch = x[batch], y[batch]
            err = 1 - y_batch * self.predict(x_batch, True)
            if np.max(err) <= 0:
                continue
            mask = err > 0
            delta = lr * c * y_batch[mask]
            self._w += np.mean(delta[..., None] * x_batch[mask], axis=0)
            self._b += np.mean(delta)
            if plot:
                plt.clf()
                plt.title(str(_))
                plt.scatter(axis[0], axis[1], c=labels)
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                xf, yf, base_matrix = get_base(nx, ny)
                z = decision_function(base_matrix).reshape((nx, ny))
                plt.contour(xf, yf, z, levels=[0])
                plt.pause(0.01)

def test0():
    from carefree.util import gen_two_clusters
    x, y = list(), list()
    for line in open('D:/Workspace/Study/MLiA/chapter06/data.txt'):
        lineList = line.strip().split('\t')
        x.append(list(map(float, lineList[:2])))
        y.append(int(lineList[-1]))
    x, y = gen_two_clusters()
    svm = LinearSVM()
    svm.fit(x, y, plot=True)
    print('Accuracy: {}'.format((svm.predict(x) == y).mean() * 100))

def test1():
    from carefree.util import gen_two_clusters
    x, y = list(), list()
    for line in open('D:/Workspace/Study/MLiA/chapter06/data.txt'):
        lineList = line.strip().split('\t')
        x.append(list(map(float, lineList[:2])))
        y.append(int(lineList[-1]))
    # x, y = gen_two_clusters()
    svm = LinearSVM2()
    svm.fit(x, y, plot=True)
    print('Accuracy: {}'.format((svm.predict(x) == y).mean() * 100))
    plt.waitforbuttonpress()

def test2():
    from carefree.util import gen_two_clusters
    x, y = list(), list()
    for line in open('D:/Workspace/Study/MLiA/chapter06/data.txt'):
        lineList = line.strip().split('\t')
        x.append(list(map(float, lineList[:2])))
        y.append(int(lineList[-1]))
    # x, y = gen_two_clusters()
    svm = LinearSVM3()
    svm.fit(x, y, plot=True)
    print('Accuracy: {}'.format((svm.predict(x) == y).mean() * 100))
    plt.waitforbuttonpress()

if __name__ == '__main__':
    test1()
