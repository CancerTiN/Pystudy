import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    inst = Perceptron()
    inst.fit(X, y)
    plt.waitforbuttonpress()
