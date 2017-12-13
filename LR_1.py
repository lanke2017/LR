import numpy as np
import matplotlib.pyplot as plt

def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    neg = data[:, 2] == 0
    pos = data[:, 2] == 1

    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True)

if __name__ == '__main__':

    data = np.loadtxt('data1.txt', delimiter=',')
    X = np.c_[np.ones((data.shape[0], 1)), data[:, 0:2]]
    y = np.c_[data[:, 2]]


