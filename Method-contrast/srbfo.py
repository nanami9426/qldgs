'''
牛奔,毕莹,郭晨.结构重组的细菌觅食优化算法及其在投资组合问题上的应用[J].中国管理科学,2014,22(S1):205-211.DOI:10.16381/j.cnki.issn1003-207x.2014.s1.040.
'''


import numpy as np
from numpy.random import rand
from Function import Fun

def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
    return X

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0
    return Xbin

def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # Parameters
    ub = 1
    lb = 0
    thres    = 0.5
    dim = np.size(xtrain, 1)
    S = opts['N']    # 种群规模
    Sr = S // 2
    max_iter = opts['T']    # 迭代次数
    Nc = max_iter    # 趋化步骤数
    Fre = 5    # 繁殖频率
    Fed = 3    # 驱散频率
    C = 0.1    # 步长数
    Ns = dim//Nc    # 游泳步数
    Ped = 0.25    # 迁徙概率

    # Dimension
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X = init_position(lb, ub, S, dim)
    # Binary conversion
    X = binary_conversion(X, thres, S, dim)

    # Pre
    fit = np.zeros([S, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
    Xpb = np.zeros([S, dim], dtype='float')
    fitP = float('inf') * np.ones([S, 1], dtype='float')
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    while t < max_iter:
        # Fitness
        for i in range(S):
            fit[i, 0] = Fun(xtrain, xvalid, ytrain, yvalid, X[i, :], opts)
            if fit[i, 0] < fitP[i, 0]:
                Xpb[i, :] = X[i, :]
                fitP[i, 0] = fit[i, 0]
            if fitP[i, 0] < fitG:
                Xgb[0, :] = Xpb[i, :]
                fitG = fitP[i, 0]
        # Store result
        fit_mean = np.mean(fitP)
        X = Xpb.copy()
        curve[0, t] = fitG.copy()
        # print("Iteration:", t + 1)
        # print("Best (SRBFO):", curve[0,t])
        t += 1
        # 趋化：游动和翻转
        if fit[i, 0] < fit_mean:
            X[i, :] = 1 - X[i, :]  # 翻转运动向量
        else:
            # 生成0到dim之间的随机整数
            random_integers = np.random.randint(0, dim, size=(Ns))  # randint默认不包括结束值
            randoms = np.random.rand(Ns)
            for m in range(Ns):
                X[i, random_integers[m]] += C * (randoms[m] - 0.5)  # 游动，随机增加或减少
        # Binary conversion
        Xbin = binary_conversion(X, thres, S, dim)
        # Fitness
        for i in range(S):
            fit[i, 0] = Fun(xtrain, xvalid, ytrain, yvalid, Xbin[i, :], opts)
            if fit[i, 0] < fitP[i, 0]:
                Xpb[i, :] = X[i, :]
                fitP[i, 0] = fit[i, 0]
            if fitP[i, 0] < fitG:
                Xgb[0, :] = Xpb[i, :]
                fitG = fitP[i, 0]
        X = Xbin.copy()

        # 繁衍复制
        if t % Fre == 0:
            sorted_indices = np.argsort(fit[:, 0])  # 评估适应度并排序
            selected_indices = sorted_indices[:Sr]  # 选择适应度最低的一半细菌索引
            better_X = X[selected_indices]  # 保留适应度最低的一半细菌
            better_fit = fit[selected_indices]
            new_X = np.vstack((better_X, better_X))
            new_fit = np.vstack((better_fit, better_fit))
            X = new_X.copy()  # 更新X和fit
            fit = new_fit.copy()

        # 消除和扩散/迁徙
        if t % Fed == 0:
            for i in range(S):
                if rand() < Ped:
                    Xnew = init_position(lb, ub, 1, dim)
                    Xnew = binary_conversion(Xnew, thres, 1, dim)
                    X[i, :] = Xnew

    # Best feature subset
    Gbin = Xgb[0, :]
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    # Create dictionary
    srbfo_data = {'sf': Gbin, 'c': curve, 'nf': num_feat}

    return srbfo_data