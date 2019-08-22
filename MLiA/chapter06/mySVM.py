import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

X, y = list(), list()
for line in open('data.txt'):
    lineList = line.strip().split('\t')
    X.append(list(map(float, lineList[:2])))
    y.append(int(lineList[-1]))

class Perceptron:
    def __init__(self):
        self._w = self._b = None

    def fit(self, x, y, lr=0.01, epoch=1000):
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylim([-1, 6])
        ax.set_xlim([-1, 6])
        x1c_pos, x2c_pos, x1c_neg, x2c_neg = list(), list(), list(), list()
        for xcs, yc in zip(X, y):
            if yc == 1:
                x1c_pos.append(xcs[0])
                x2c_pos.append(xcs[1])
            else:
                x1c_neg.append(xcs[0])
                x2c_neg.append(xcs[1])
        else:
            ax.scatter(x1c_pos, x2c_pos, c='red', marker='s')
            ax.scatter(x1c_neg, x2c_neg, c='blue')
        # x.shape == (20, 2) y.shape == (20,)
        x = np.asarray(x, np.float32)
        y = np.asarray(y, np.float32)
        # w.shape == (2,)
        w = np.zeros(x.shape[1])
        b = 0.0
        for _ in range(epoch):
            # y_pred -> wx+b
            # y_pred.shape == (20,)
            y_pred = np.dot(x, w) + b
            # -y_pred * y -> -y(wx+b)
            # np.maximum(0, -y_pred * y) -> L(x,y)
            max_idx = np.argmax(np.maximum(0, -y_pred * y))
            # y[max_idx] in [-1, 1]
            # 关键环节，不妨认为当y[max_idx]为1时，
            # 假设模型正确分类，则必有y_pred[max_idx]大于0，1乘以大于零的数结果大于0，退出循环
            # =================================================================================
            # 但是由于上一个语句的np.maximum(0, -y_pred * y)使得在max_idx处必然有-y_pred * y小于0，
            # 即np.maximum(0, -y_pred * y)将在max_idx处返回0，
            # 注意这里有一个博弈点，因为y_pred是一个包含预测分类结果的向量，
            # 当出现分类错误时，比如y[2]为1，y_pred[2]将小于0，则-y_pred[2] * y[2]的结果大于0，
            # 对比上述分类正确和分类错误的情况，可以发现当y_pred与y在对应索引中存在正负符号不同时，
            # 通过np.maximum(0, -y_pred * y)函数必定使结果向量中分类错误的索引处的结果值最大（分类正确的都为0），
            # 于是可以了解到max_idx在y_pred处对应的值一定是分类错误的（当y_pred中仍还有分类错误时），
            # 如果没有完成收敛，只有迭代完成（1000次循环）才可以使计算结束，这个break是不会被运行的
            # =================================================================================
            # break的可能仅为max_idx为0，即所有y_pred内的值和y中的符号都相同，
            # 此时np.maximum(0, -y_pred * y)将返回一个全是0的向量，用np.argmax将返回第一个索引0，
            # 因为y[0]和y_pred[0]符号相同，于是y[0] * y_pred[0]将大于0，发生break
            if y[max_idx] * y_pred[max_idx] > 0:
                break
            delta = lr * y[max_idx]
            w += delta * x[max_idx]
            b += delta

            x1p = np.arange(0, 5, 0.1)
            x2p = (-b - w[0] * x1p) / w[1]
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass

            lines = ax.plot(x1p, x2p, c='black')
            #fig.canvas.draw_idle()
            plt.pause(0.1)

class LinearSVM:
    def __init__(self):
        self._w = self._b = None

    def fit(self, x, y, c=1, lr=0.01, batch_size=128, epoch=300, plot=True):
        # plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.set_ylim([-1, 6])
        # ax.set_xlim([-1, 6])
        # x1c_pos, x2c_pos, x1c_neg, x2c_neg = list(), list(), list(), list()
        # for xcs, yc in zip(X, y):
        #     if yc == 1:
        #         x1c_pos.append(xcs[0])
        #         x2c_pos.append(xcs[1])
        #     else:
        #         x1c_neg.append(xcs[0])
        #         x2c_neg.append(xcs[1])
        # else:
        #     ax.scatter(x1c_pos, x2c_pos, c='red', marker='s')
        #     ax.scatter(x1c_neg, x2c_neg, c='blue')

        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        batch_size = min(batch_size, len(y))
        self._w = np.zeros(x.shape[1])
        self._b = 0.
        n = 1
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
            batch = np.random.choice(len(x), batch_size)
            x_batch, y_batch = x[batch], y[batch]

            # 当分类正确时，L > 0
            # 当分类错误时，L < 0
            err = 1 - y_batch * self.predict(x_batch, True)
            # L = y_batch * self.predict(x_batch, True)
            # err = 1 - L
            # 如果np.max(err)小于等于0，说明L的所有值都大于等于1，
            # 说明batch向量的分类都正确，直接迭代到下一个循环
            if np.max(err) <= 0:
                continue
            # mask为挑选出batch中分类错误的bool索引
            mask = err > 0
            delta = lr * c * y_batch[mask]
            self._w += np.mean(delta[..., None] * x_batch[mask], axis=0)
            self._b += np.mean(delta)

            # x1p = np.arange(-1, 6, 0.1)
            # x2p = (-self._b - self._w[0] * x1p) / self._w[1]
            # try:
            #     ax.lines.remove(lines[0])
            # except Exception:
            #     pass
            # lines = ax.plot(x1p, x2p, c='black')
            # plt.title(str(n))
            # n += 1
            # plt.pause(0.001)
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

    def predict(self, x, raw=False):
        x = np.asarray(x, np.float32)
        y_pred = x.dot(self._w) + self._b
        if raw:
            return y_pred
        return np.sign(y_pred).astype(np.float32)


if __name__ == '__main__':
    # inst = Perceptron()
    inst = LinearSVM()
    inst.fit(X, y)
    plt.waitforbuttonpress()
