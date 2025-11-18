import copy
import numpy as np
from Function import Fun
from numpy.random import rand
import math
import warnings
import random
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings("ignore")

def init_position(lb, ub, N, dim):
    # Generate a matrix of random values in [0, 1]
    X_rand = np.random.rand(N, dim)
    # Scale and shift to the range [lb[0, d], ub[0, d]]
    X = lb[0, :] + (ub[0, :] - lb[0, :]) * X_rand
    return X


def init_velocity(lb, ub, N, dim):
    # Compute Vmax and Vmin using vectorized operations
    Vmax = (ub[0, :] - lb[0, :]) / 2
    Vmin = -Vmax
    # Generate a matrix of random values in [0, 1]
    V_rand = np.random.rand(N, dim)
    # Scale and shift to the range [Vmin, Vmax]
    V = Vmin + (Vmax - Vmin) * V_rand
    return V, Vmax[np.newaxis, :], Vmin[np.newaxis, :]

def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    return x

def initialize_population(pop_size, num_features, max_length):
    population = []
    for _ in range(pop_size):
        # 随机初始化长度
        length = np.random.randint(2, size=length)
        particle = np.random.randint(2, size=length)
        population.append({'position': particle, 'velocity': np.zeros(length), 'best_position': particle.copy()})
    return population

def inittial_pos(maxlen):
    DivX = np.random.rand(maxlen)
    DivV = -1 + 2 * np.random.rand( maxlen)
    return DivX, DivV

def binary_conversion(X, maxLen,thres, dim):
    Xbin = np.zeros(dim, dtype='int')  # Initialize Xbin with the desired dimension
    Xbin[:maxLen] = (X[:maxLen] > thres).astype(int)
    return Xbin

def compute_avg(arr, c):
    # 计算总数并进行分块
    n = len(arr)
    # 计算分块数量，最后一个块可能不足c个元素
    num_chunks = (n + c - 1) // c
    # 使用reshape来将数组分成指定大小的块，如果最后一块不足c个元素，它会自动填充
    chunks = np.array_split(arr, num_chunks)
    # 对每块计算平均值
    avg_values = [np.mean(chunk) for chunk in chunks]
    return avg_values

# 假设xtrain, xvalid, ytrain, yvalid已经存在
# 这里我们会使用mutual_info_classif来计算特征和标签之间的互信息，
# 并根据互信息生成对称不确定性以及对应的特征排名。

# 计算熵
def entropy(labels):
    # 标签的概率分布
    label_prob = np.bincount(labels) / len(labels)
    # 计算熵
    return -np.sum(label_prob * np.log2(label_prob + 1e-9))  # 防止log(0)的情况

# 计算训练集中特征与标签的对称不确定性
def calculate_symmetrical_uncertainty(X_train, y_train):
    # 确保 labels 是整数类型
    labels = y_train.astype(np.int64)  # 明确指定转换为 int64 类型
    # 或者：
    # labels = np.round(y_train).astype(np.int64)  # 如果 y_train 包含浮点数，先四舍五入

    # 计算特征与目标之间的互信息
    mi = mutual_info_classif(X_train, labels)

    # 计算每个特征与目标的熵
    entropy_X = np.array([entropy(X_train[:, i]) for i in range(X_train.shape[1])])
    entropy_y = entropy(labels)

    # 对称不确定性：2 * 互信息 / (特征的熵 + 标签的熵)
    su = 2 * mi / (entropy_X + entropy_y)
    return su

def fs(xtrain, xvalid, ytrain, yvalid, opts):
    N = opts['N']
    popSize = 100
    beishu = popSize/N
    T = opts['T']
    maxT = int(T/beishu)
    NbrDiv = 10
    thres = 0.6
    DivSize = int(np.ceil(popSize/NbrDiv))
    Vmax = 1
    Vmin = -1
    Xmax = 1
    Xmin = 0
    # Dimension
    dim = np.size(xtrain, 1)
    maxLength = dim
    X = []
    V = []
    XPBest = []
    DivLabel = np.zeros(popSize)
    DivMaxLen = np.zeros(popSize)
    fitPBest = np.ones(popSize)
    # Div_mean_fit = np.ones(popSize)
    fitG = 1
    Xgb = np.zeros(dim)
    Pc = np.zeros(popSize)
    j = 0
    # w = 0.5
    c = 1.49445
    curve = np.zeros([1, maxT], dtype='float')
    # # 计算对称不确定性
    # su_values = calculate_symmetrical_uncertainty(xtrain, ytrain)
    # # 生成特征排名
    # feature_rankings = np.argsort(su_values)[::-1]  # 从高到低排序

    feature_rankings = np.array([p for p in range(dim)])

    # 生成种群群体
    for i in range(popSize):
        if i % DivSize == 0:
            j = j + 1
        ParLen = int(np.floor(maxLength * j / NbrDiv))
        DivX, DivV = inittial_pos(ParLen)
        X.append(DivX.copy())
        V.append(DivV.copy())
        DivLabel[i] = j
        DivMaxLen[i] = ParLen
        Xbin = binary_conversion(DivX, ParLen, thres, dim)
        selected_indices = np.where(Xbin == 1)[0]
        Xbin_ = np.zeros_like(Xbin)
        Xbin_[feature_rankings[selected_indices]] = 1
        fit = Fun(xtrain, xvalid, ytrain, yvalid, Xbin_, opts)
        XPBest.append(DivX)
        fitPBest[i] = fit
        if fit < fitG:
            Xgb = Xbin.copy()
            fitG = fit
    rankPop = np.argsort(fitPBest)
    exemplars = []
    for i in range(popSize):
        Pc[i] = 0.05 + 0.45*np.exp(10*(rankPop[i] - 1)/(popSize-1))/(np.exp(10) - 1)
        L = int(DivMaxLen[i])
        exemplar = np.zeros(L)
        for d in range(L):
            r = np.random.rand()
            if r >= Pc[i]:
                exemplar[d] = i
            else:
                index_max = np.where(np.array(DivMaxLen) > d)[0]
                # print(f'{index_max}')
                index_to_remove = np.where(index_max == i)[0][0]  # 找到元素3的索引
                # print(f'{index_to_remove}')
                index_max1 = np.delete(index_max, index_to_remove)
                # print(f'{index_max1}')
                if index_max1.shape[0] == 1:
                    exemplar[d] = index_max1[0]
                    break
                elif index_max1.shape[0] == 0:
                    exemplar[d] = i
                    break
                selected_numbers = random.sample(index_max1.tolist(), 2)
                if fitPBest[selected_numbers[0]] < fitPBest[selected_numbers[1]]:
                    exemplar[d] = selected_numbers[0]
                else:
                    exemplar[d] = selected_numbers[1]
        exemplars.append(exemplar.copy())

    t = 0
    PBestI_T = np.zeros(popSize)
    PBestI_F = np.zeros(popSize)
    alpha = 7
    beta = 9
    gBestI_T = 0
    while t < maxT:
        w = 0.9 - 0.5 * t / maxT
        gBestI_F = 0
        for i in range(popSize):
            # 更新Pc
            if(PBestI_F[i] == 1):
                rankPop = np.argsort(fitPBest)
                Pc[i] = 0.05 + 0.45 * np.exp(10 * (rankPop[i] - 1) / (popSize - 1)) / (np.exp(10) - 1)
                L = int(DivMaxLen[i])
                exemplar = np.zeros(L)
                for d in range(L):
                    r = np.random.rand()
                    if r >= Pc[i]:
                        exemplar[d] = i
                    else:
                        index_max = np.where(np.array(DivMaxLen) > d)[0]
                        index_to_remove = np.where(index_max == i)[0][0]  # 找到元素3的索引
                        index_max1 = np.delete(index_max, index_to_remove)
                        if index_max1.shape[0] == 1:
                            exemplar[d] = index_max1[0]
                            break
                        elif index_max1.shape[0] == 0:
                            exemplar[d] = i
                            break
                        selected_numbers = random.sample(index_max1.tolist(), 2)
                        if fitPBest[selected_numbers[0]] < fitPBest[selected_numbers[1]]:
                            exemplar[d] = selected_numbers[0]
                        else:
                            exemplar[d] = selected_numbers[1]
                exemplars[i] = exemplar.copy()
                PBestI_F[i] = 0
            # 更新速度和位置
            L = int(DivMaxLen[i])
            for d in range(L):

                V[i][d] = w * V[i][d] + c * np.random.rand() * (X[int(exemplars[i][d])][d] - X[i][d])
                if V[i][d] > Vmax:
                    V[i][d] = Vmax
                elif V[i][d] < Vmin:
                    V[i][d] = Vmin
                X[i][d] = X[i][d] + V[i][d]
                if X[i][d] > Xmax:
                    X[i][d] = Xmax
                elif X[i][d] < Xmin:
                    X[i][d] = Xmin
            Xbin = binary_conversion(X[i], int(DivMaxLen[i]), thres, dim)
            selected_indices = np.where(Xbin == 1)[0]
            Xbin_ = np.zeros_like(Xbin)
            Xbin_[feature_rankings[selected_indices]] = 1
            fit = Fun(xtrain, xvalid, ytrain, yvalid, Xbin_, opts)
            if fit < fitPBest[i]:
                fitPBest[i] = fit
                XPBest[i] = X[i].copy()
                PBestI_T[i] = 0
            PBestI_T[i] = PBestI_T[i] + 1
            if PBestI_T[i] % alpha == 0:
                PBestI_F[i] = 1

            if fit < fitG:
                gBestI_F = 1
                fitG = fit
                Xgb = Xbin.copy()
        if gBestI_F == 1:
            gBestI_T = 0
            gBestI_F = 0

        gBestI_T = gBestI_T + 1
        # 长度机制变换
        if gBestI_T % beta == 0:
            divFit_arr = compute_avg(fitPBest, DivSize)
            divBestIndex = np.argmin(divFit_arr)
            BestLen = DivMaxLen[divBestIndex * DivSize]
            if BestLen == maxLength:
                break
            k = 0
            for i in range(popSize):
                if i % DivSize == 0:
                    k = k + 1
                newLen = int(np.floor(BestLen * k / NbrDiv))
                if DivMaxLen[i] < newLen:
                    addLen = newLen - DivMaxLen[i]
                    DivMaxLen[i] = newLen
                    random_list = np.array([random.random() for _ in addLen])
                    X[i] = np.append(X[i], random_list)
                    random_list = np.array([-1 + 2 * random.random() for _ in addLen])
                    X[i] = np.append(V[i], random_list)
                    Xbin = binary_conversion(X[i], newLen, thres, dim)
                    selected_indices = np.where(Xbin == 1)[0]
                    Xbin_ = np.zeros_like(Xbin)
                    Xbin_[feature_rankings[selected_indices]] = 1
                    fit = Fun(xtrain, xvalid, ytrain, yvalid, Xbin_, opts)
                    if fit < fitPBest[i]:
                        fitPBest[i] = fit
                        XPBest[i] = X[i].copy()
                    if fit < fitG:
                        Xgb = Xbin.copy()
                        fitG = fit
                elif DivMaxLen[i] > newLen:
                    # deleteLen = DivMaxLen[i] - newLen
                    X[i] = X[i][0:newLen]
                    V[i] = V[i][0:newLen]
                    DivMaxLen[i] = newLen
                    Xbin = binary_conversion(X[i], newLen, thres, dim)
                    selected_indices = np.where(Xbin == 1)[0]
                    Xbin_ = np.zeros_like(Xbin)
                    Xbin_[feature_rankings[selected_indices]] = 1
                    fit = Fun(xtrain, xvalid, ytrain, yvalid, Xbin_, opts)
                    if fit < fitPBest[i]:
                        fitPBest[i] = fit
                        XPBest[i] = X[i].copy()
                    if fit < fitG:
                        Xgb = Xbin.copy()
                        fitG = fit
                PBestI_F[i] = 1
                PBestI_T[i] = 0

        curve[0, t] = fitG
        t = t + 1
    Gbin = Xgb
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    # Create dictionary
    expanded_curve = np.repeat(curve, 5, axis=1)
    # 计算需要等差选取的数据点个数
    num_points = opts['T']
    # 等差选取数据点的索引
    indices = np.round(np.linspace(0, np.size(expanded_curve, 1) - 1, num_points)).astype(int)
    # 从Cost中选取数据填入curve
    curve0 = expanded_curve[:, indices]
    # 将curve变形为所需的大小
    curve0 = curve0.reshape(1, -1)
    # 现在curve0中包含了从expanded_curve中等差选取的数据
    vlpso_data = {'sf': Gbin, 'c': curve0, 'nf': num_feat}
    return vlpso_data