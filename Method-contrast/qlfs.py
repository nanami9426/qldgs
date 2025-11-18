import numpy as np
from Function import Fun
from numpy.random import rand
import math
import warnings
"Dynamic feature selection algorithm based on Q-learning mechanism"

def qlfs(train_data, labels):
    """
    Q-learning based Feature Selection (QLFS) algorithm.

    :param train_data: mxn numpy array, where m is the number of samples and n is the number of features.
    :param labels: 1-dimensional numpy array with length m, containing the labels for each sample.
    :return: ranked: 1-dimensional numpy array with the ranking of features.
    """
    # 转置特征矩阵，使其成为(n, m)的形状
    train_data = train_data.T

    # 特征数量和样本数量
    feature = train_data.shape[0]
    number = train_data.shape[1]

    # 初始化状态矩阵，添加一列用于时间步长
    state = np.zeros((feature + 1, number))
    state[0, :] = 1  # 第一列为全1，用于偏置项

    # 填充状态矩阵，将特征数据填充到状态矩阵中
    state[1:, :] = train_data

    # 初始化权重矩阵和排名数组
    c = np.unique(labels).size  # 计算类别数
    w = np.zeros((c, feature + 1))  # 权重矩阵，添加一列用于偏置项
    ranked = np.zeros(feature)  # 特征排名数组

    # 学习率
    alpha = 30

    # 主循环
    for t in range(1):  # 根据需要设置迭代次数
        w = np.zeros((c, feature + 1))  # 重置权重矩阵

        for k in range(1000):  # 外部循环，可以设置为需要的迭代次数
            w0 = w.copy()  # 保存当前权重矩阵状态

            for i1 in range(number):  # 内部循环
                # 随机选择动作，探索概率为0.8
                if np.random.rand() > 0.8:
                    v = np.dot(w, state[:, i1])  # 计算所有类别的判别分数
                    d = np.argmax(v)  # 选择最大判别分数的类别
                else:
                    d = np.random.randint(0, c)  # 随机选择类别

                # 计算奖励
                R = 1 if d == labels[i1] else -1
                if i1 < number - 1:
                    R += 0.1 * np.max(np.dot(w, state[:, i1 + 1]))

                # 更新权重
                for z in range(c):
                    if z == d:
                        temp = (R - np.dot(w[z, :], state[:, i1])) * state[:, i1]
                        w[z, :] += (1 / (alpha * ((i1 + 1) ** 0.2))) * temp
                        break

            # 如果权重矩阵变化很小，则提前结束循环
            if np.linalg.norm(w - w0) < 0.000001:
                break

        # 特征选择和排名逻辑
        # 这里需要根据QLFS算法的逻辑来实现特征的删除和排名
        # 以下是一个示例逻辑，可能需要根据实际算法调整
        w_abs = np.abs(w)
        w_abs_sum = np.sum(w_abs, axis=0)[:-1]
        ranked[w_abs_sum.argsort()[::-1]] = np.arange(feature)  # 根据权重的绝对值和进行排名

    return ranked


def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # x = np.concatenate((xtrain, xvalid), axis=0)
    # y = np.concatenate((ytrain, yvalid), axis=0)
    x = xtrain.copy()
    y = ytrain.copy()
    rank = qlfs(x, y)
    dim = np.size(xtrain, 1)
    N = opts['N']
    max_iter = opts['T']
    value_num = int(dim * 0.4)
    curve = np.zeros([1, value_num], dtype='float')
    Xpb = np.zeros([value_num, dim], dtype='int')
    # fitP = float('inf') * np.ones([max_iter, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='int')
    fitG = float('inf')
    for i in range(value_num):
        sf = rank[:value_num - i - 1]
        Xpb[i, sf.astype(int)] = 1
        fitP = Fun(xtrain, xvalid, ytrain, yvalid, Xpb[i, :], opts)
        if fitP <= fitG:
            Xgb[0, :] = Xpb[i, :]
            fitG = fitP
        curve[0, i] = fitG.copy()
    num_feat = np.sum(Xgb[0, :])
    expanded_vector = np.zeros((1,N * max_iter))

    # 计算每个元素需要重复的次数，如果n是max_n的因数，则每个元素重复n次；否则，重复次数为n-1
    repeat_times = (N * max_iter // value_num) + 1

    # 填充扩展后的向量
    for i in range(value_num):
        # 将当前元素值重复指定次数并添加到扩展向量中
        expanded_vector[0, i * repeat_times:(i + 1) * repeat_times] = curve[0, i]

    num_points = opts['T']
    # 等差选取数据点的索引
    indices = np.round(np.linspace(0, N * max_iter - 1, num_points)).astype(int)
    # 从Cost中选取数据填入curve
    expanded_vector = expanded_vector[:, indices]
    # 将curve变形为所需的大小
    expanded_vector = expanded_vector.reshape(1, -1)
    inf_data = {'sf': Xgb, 'c': expanded_vector, 'nf': num_feat}

    return inf_data

