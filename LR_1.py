import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


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


def sigmoid(z):
    # 定义sigmoid函数
    # from scipy.special import expit
    # expit(1.2)
    return(1 / (1 + np.exp(-z)))


def costFunction(theta, X, y):
    # 定义损失函数
    #
    m = y.size
    h = sigmoid(X.dot(theta))
    h1 = sigmoid(np.dot(X, theta))
    J = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))
    J1 = -1.0 * (1.0 / m) * (np.dot(np.log(h1), y) + np.dot(np.log(1-h1), (1-y)))
    if np.isnan(J[0]):
        return (np.inf)
    return J[0]


if __name__ == '__main__':

    data = np.loadtxt('data1.txt', delimiter=',')
    X = np.c_[np.ones((data.shape[0], 1)), data[:, 0:2]]  # 取data所有特征，第一列全为1，二三列为原特征
    y = np.c_[data[:, 2]]  # 所有特征（两列）
    #
    plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')
    plt.show()
    #
    initial_theta = np.zeros(X.shape[1])
    cost = costFunction(initial_theta, X, y)


