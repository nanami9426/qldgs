import numpy as np
from numpy.random import rand
from Function import Fun
import math

def init_position(lb, ub, N, dim, percentage):
    X = np.full((N, dim), lb)
    num_ub = int(dim * percentage)
    for i in range(N):
        indices = np.random.choice(dim, num_ub, replace=False)
        X[i, indices] = ub
    return X

def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # opts['split'] = 5
    opts['func'] = 1
    ub = 1
    lb = 0
    N = opts['N']
    max_iter = opts['T']
    dim = np.size(xtrain, 1)
    per = 2/3  # per应该大于等于0.5
    URF_max = 1 - per
    dim_level = int(math.log10(dim))
    URF_min = round(1 / (10 ** (dim_level + 1)), dim_level + 1)
    alpha = 5  # 控制曲线陡峭程度的参数(UR)
    factor = 0
    matrix = np.arange(max_iter * N).reshape((1, max_iter * N))
    # Dimensionality reduction factor
    URF = (1 / (1 + np.exp(-alpha * (-2 * (matrix + 1) / (max_iter * N) + 1)))) * (URF_max - URF_min) + URF_min
    UR = np.ceil(URF * dim)
    # Combined Learning Factor
    CLF = per - ((1 / (1 + np.exp(-alpha * (-2 * (matrix + 1) / (max_iter * N) + 1)))) * (per - URF) + URF)

    # Initialize position
    X = init_position(lb, ub, N, dim, per)

    # Pre
    fit = np.zeros([N, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
    Xpb = np.zeros([N, dim], dtype='float')
    fitP = float('inf') * np.ones([N, 1], dtype='float')
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    # Fitness
    for i in range(N):
        fit[i, 0] = Fun(xtrain, xvalid, ytrain, yvalid, X[i, :], opts)
        if fit[i, 0] < fitP[i, 0] or (fit[i, 0] == fitP[i, 0] and np.sum(X[i, :]) < np.sum(Xpb[i, :])):
            Xpb[i, :] = np.copy(X[i, :])
            fitP[i, 0] = np.copy(fit[i, 0])
        if fitP[i, 0] < fitG or (fitP[i, 0] == fitG and np.sum(Xpb[i, :]) < np.sum(Xgb[0, :])):
            Xgb[0, :] = np.copy(Xpb[i, :])
            fitG = np.copy(fitP[i, 0])
            # bestX = i

    while t < max_iter:
        for i in range(N):
            if UR[0,factor] < np.sum(X[i, :]):
                # 降维操作
                X_select = np.where(X[i, :] == 1)[0]
                # 随机选择要剔除的索引
                indices_to_remove = np.random.choice(X_select, int(UR[0,factor]), replace=False)
                X[i, indices_to_remove] = lb
                if np.sum(X[i, :]) < 1:
                    X[i, :] = np.copy(Xgb[0, :])
            else:
                # 组合效应1
                X_select = np.where(X[i, :] == 1)
                Xgb_select = np.where(Xgb[0, :] == 1)
                # 找出 Xgb_select 中有而 X_select 中没有的数
                difference = np.setdiff1d(Xgb_select, X_select)
                if len(difference) == 0:
                    column_sum = np.sum(Xpb, axis=0)
                    X_select = np.where(column_sum > 1)[0]
                    difference = np.setdiff1d(X_select, Xgb_select)
                if len(difference) > 0:
                    add_num = math.ceil(CLF[0,factor] * np.size(difference, 0))
                    indices_to_add = np.random.choice(difference.flatten(), add_num, replace=False)
                    X[i, indices_to_add] = ub
                else:
                    X[i, :] = init_position(lb, ub, 1, dim, 1-per)
                # # 组合效应2（效果低于V1）
                # if i == bestX:
                #     column_sum = np.sum(Xpb, axis=0)
                #     X_select = np.where(column_sum > 1)[0]
                #     Xgb_select = np.where(Xgb[0, :] == 1)
                #     difference = np.setdiff1d(X_select, Xgb_select)
                #     add_num = math.ceil(1 * np.size(difference, 0))
                # else:
                #     excellentX = np.where(fitP[:, 0] <= fitP[i, 0])[0]
                #     learnX = np.random.choice(excellentX)
                #     Xexc_select = np.where(Xpb[learnX, :] == 1)
                #     X_select = np.where(X[i, :] == 1)
                #     difference = np.setdiff1d(Xexc_select, X_select)
                #     add_num = math.ceil(per * np.size(difference, 0))
                # indices_to_add = np.random.choice(difference.flatten(), add_num, replace=False)
                # X[i, indices_to_add] = ub
            factor += 1

        # Fitness
        for i in range(N):
            fit[i, 0] = Fun(xtrain, xvalid, ytrain, yvalid, X[i, :], opts)
            if fit[i, 0] < fitP[i, 0] or (fit[i, 0] == fitP[i, 0] and np.sum(X[i, :]) < np.sum(Xpb[i, :])):
                Xpb[i, :] = np.copy(X[i, :])
                fitP[i, 0] = np.copy(fit[i, 0])
            if fitP[i, 0] < fitG or (fitP[i, 0] == fitG and np.sum(Xpb[i, :]) < np.sum(Xgb[0, :])):
                Xgb[0, :] = np.copy(Xpb[i, :])
                fitG = np.copy(fitP[i, 0])
                # bestX = i
        X = np.copy(Xpb)
        curve[0, t] = fitG.copy()
        t += 1

    # Best feature subset
    num_feat = np.sum(Xgb)
    pos = np.asarray(range(0, dim))
    sf_index = pos[Xgb[0, :] == 1]
    elh_data = {'sf': Xgb, 'c': curve, 'nf': num_feat}
    return elh_data