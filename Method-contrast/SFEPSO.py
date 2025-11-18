import copy

import numpy as np
from Function import Fun
from numpy.random import rand
import math
import warnings
warnings.filterwarnings("ignore")

def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
    return X

def init_velocity(lb, ub, N, dim):
    V = np.zeros([N, dim], dtype='float')
    Vmax = np.zeros([1, dim], dtype='float')
    Vmin = np.zeros([1, dim], dtype='float')
    # Maximum & minimum velocity
    for d in range(dim):
        Vmax[0, d] = (ub[0, d] - lb[0, d]) / 2
        Vmin[0, d] = -Vmax[0, d]
    for i in range(N):
        for d in range(dim):
            V[i, d] = Vmin[0, d] + (Vmax[0, d] - Vmin[0, d]) * rand()
    return V, Vmax, Vmin

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0
    return Xbin

def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    return x

def fs(xtrain, xvalid, ytrain, yvalid, opts):
    Input = xtrain
    Target = ytrain
    UR = 0.3
    UR_Max = 0.3
    UR_Min = 0.001
    Max_Run = 1
    Run = 1
    Max_FEs = opts['T'] * opts['N']
    Cost = np.zeros([Max_Run, Max_FEs])
    FN = np.zeros([Max_Run, Max_FEs])
    arr = np.random.randint(0, 2, 10)
    pso_start = 2000
    pso_interval = 1000

    while (Run <= Max_Run):
        EFs = 1

        X = np.random.randint(0, 2, np.size(xtrain, 1))  # Initialize an Individual X
        Fit_X = Fun(xtrain, xvalid, ytrain, yvalid, X, opts)  # Calculate the Fitness of X
        Nvar = np.size(xtrain, 1)  # Number of Features in Dataset

        while (EFs <= Max_FEs):

            X_New = np.copy(X)
            # Non-selection operation:
            U_Index = np.where(X == 1)  # Find Selected Features in X
            NUSF_X = np.size(U_Index, 1)  # Number of Selected Features in X
            UN = math.ceil(UR * Nvar)  # The Number of Features to Unselect: Eq(2)
            # SF=randperm(20,1)                             # The Number of Features to Unselect: Eq(4)
            # UN=ceil(rand*Nvar/SF);                        # The Number of Features to Unselect: Eq(4)
            K1 = np.random.randint(0, NUSF_X,
                                   UN)  # Generate UN random number between 1 to the number of slected features in X
            res = np.array([*set(K1)])
            res1 = np.array(res)
            K = U_Index[0][[res1]]  # K=index(U)
            X_New[K] = 0  # Set X_New (K)=0
            # Selection operation:
            if np.sum(X_New) == 0:
                S_Index = np.where(X == 0)  # Find non-selected Features in X
                NSF_X = np.size(S_Index, 1)  # Number of non-selected Features in X
                SN = 1  # The Number of Features to Select
                K1 = np.random.randint(0, NSF_X, SN)  # Generate SN random number between 1 to the number of non-selected features in X
                res = np.array([*set(K1)])
                res1 = np.array(res)
                K = S_Index[0][[res1]]
                X_New = np.copy(X)
                X_New[K] = 1  # Set X_New (K)=1

            Fit_X_New = Fun(xtrain, xvalid, ytrain, yvalid, X_New, opts)  # Calculate the Fitness of X_New

            if Fit_X_New < Fit_X:
                X = np.copy(X_New)
                Fit_X = Fit_X_New

            UR = (UR_Max - UR_Min) * ((Max_FEs - EFs) / Max_FEs) + UR_Min  # Eq(3)
            Cost[Run - 1, EFs - 1] = Fit_X
            EFs = EFs + 1
            # 挑选的特征索引
            X_sfe_sf = np.where(X == 1)[0]

            #  pso
            if EFs > pso_start and Cost[Run - 1, EFs - 2] - Cost[Run - 1, EFs - pso_interval - 2]  == 0 and np.sum(X) > 2 :
                xtrain_pso = copy.copy(xtrain[:, X == 1])
                xvalid_pso = copy.copy(xvalid[:, X == 1])
                # Parameters
                ub = 1
                lb = 0
                thres = 0.5
                w = 0.9  # inertia weight
                c1 = 2  # acceleration factor
                c2 = 2  # acceleration factor
                N = 50
                # Dimension
                dim = np.size(xtrain_pso, 1)
                if np.size(lb) == 1:
                    ub = ub * np.ones([1, dim], dtype='float')
                    lb = lb * np.ones([1, dim], dtype='float')
                # Initialize position & velocity
                X_pso = init_position(lb, ub, N, dim)
                V, Vmax, Vmin = init_velocity(lb, ub, N, dim)
                # Pre
                fit = np.zeros([N, 1], dtype='float')
                Xgb = np.ones([1, dim], dtype='float')
                fitG = Fit_X.copy()
                Xpb = np.ones([N, dim], dtype='float')
                fitP = Fit_X.copy() * np.ones([N, 1], dtype='float')

                while EFs <= Max_FEs - N:
                    # Binary conversion
                    Xbin = binary_conversion(X_pso, thres, N, dim)
                    # Fitness
                    for i in range(N):
                        fit[i, 0] = Fun(xtrain_pso, xvalid_pso, ytrain, yvalid, Xbin[i, :], opts)
                        if fit[i, 0] < fitP[i, 0]:
                            Xpb[i, :] = X_pso[i, :]
                            fitP[i, 0] = fit[i, 0]
                        if fitP[i, 0] < fitG:
                            Xgb[0, :] = Xpb[i, :]
                            fitG = fitP[i, 0]
                        Cost[Run - 1, EFs - 1] = fitG.copy()
                        EFs = EFs + 1
                    for i in range(N):
                        for d in range(dim):
                            # Update velocity
                            r1 = rand()
                            r2 = rand()
                            V[i, d] = w * V[i, d] + c1 * r1 * (Xpb[i, d] - X_pso[i, d]) + c2 * r2 * (Xgb[0, d] - X_pso[i, d])
                            # Boundary
                            V[i, d] = boundary(V[i, d], Vmin[0, d], Vmax[0, d])
                            # Update position
                            X_pso[i, d] = X_pso[i, d] + V[i, d]
                            # Boundary
                            X_pso[i, d] = boundary(X_pso[i, d], lb[0, d], ub[0, d])

                # 找出 Xgb 中为 1 的索引
                X_pso_sf = np.where(Xgb > thres)[1]
                X_sf = X_sfe_sf[X_pso_sf]
                X = np.zeros([Nvar], dtype='float')  # 将数组X初始化为一维的
                X[X_sf] = 1  # 根据索引X_sf将对应的元素设置为1

        Gbin = X.copy()

        Run = Run + 1
    # 计算需要等差选取的数据点个数
    num_points = opts['T']
    # 等差选取数据点的索引
    indices = np.round(np.linspace(0, Max_FEs - 1, num_points)).astype(int)
    # 从Cost中选取数据填入curve
    curve = Cost[:, indices]
    # 将curve变形为所需的大小
    curve = curve.reshape(1, -1)
    # 现在curve中包含了从Cost中等差选取的数据
    sfepso_data = {'sf': Gbin, 'c': curve, 'nf': np.sum(Gbin)}
    return sfepso_data