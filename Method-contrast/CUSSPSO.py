import copy
import math

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.neighbors import KNeighborsClassifier


from skrebate import ReliefF

import numpy as np
from numpy.random import rand
from Function import Fun


# K. Chen, B. Xue, M. Zhang and F. Zhou, "Correlation-Guided Updating Strategy for Feature Selection in Classification With Surrogate-Assisted Particle Swarm Optimization,"



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

def distance(p1, p2):
        return np.sum(np.abs(p1 - p2))




def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    w = 0.9  # inertia weight
    c1 = 1.49445  # acceleration factor
    c2 = 1.49445  # acceleration factor
    Npre = opts['N']
    T = opts['T']
    curves = np.ones([1, T], dtype='float')
    if 'w' in opts:
        w = opts['w']
    if 'c1' in opts:
        c1 = opts['c1']
    if 'c2' in opts:
        c2 = opts['c2']
        # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
    gBestAcc = 1
    Nc = 2
    max_iter = (Npre * T) // 100
    N = 100

    # N = N
    # Threshold = Threshold

    # dim = feature.shape[1]
    positionPre = np.zeros((N, dim))
    positionNew = np.zeros((N, dim))
    velocityNew = np.zeros((N, dim))
    velocityPre = np.zeros((N, dim))
    pArrayNew = np.zeros((N, dim))
    pArrayPre = np.zeros((N, dim))
    accuracyNew = np.zeros(N)
    accuracyPre = np.zeros(N)
    selectDimNew = np.zeros(N)
    selectDimPre = np.zeros(N)
    fitnessPre = np.zeros(N)
    fitnessNew = np.zeros(N)
    CPosition = np.zeros((N * (Nc + 1), dim))
    CVelocity = np.zeros((N * (Nc + 1), dim))
    CFitness = np.zeros(N * (Nc + 1))
    SPosition = np.zeros((N, dim))
    SVelocity = np.zeros((N, dim))
    SFitness = np.zeros(N)
    pBest = np.zeros((N, dim))
    pBestFit = np.ones(N)
    gBest = np.zeros(dim)
    gBestArr = np.zeros(dim)
    gBestFit = 1
    probability = np.zeros(dim)
    A = 0.15
    B = 0.05
    # w = w
    # c1 = c1
    # c2 = c2
    # r1 = r1
    # r2 = r2
    # fit = np.zeros([N, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
    Xpb = np.zeros([N, dim], dtype='float')
    fitP = float('inf') * np.ones([N, 1], dtype='float')
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0
    positionNew = init_position(lb, ub, N, dim)
    X_new = binary_conversion(positionNew, thres, N, dim)
    for i in range (N):
        fitnessNew[i] = Fun(xtrain, xvalid, ytrain, yvalid, X_new[i, :], opts)
        if fitnessNew[i] < pBestFit[i]:
            pBest[i, :] = copy.copy(positionNew[i, :])
            pBestFit[i] = fitnessNew[i]
        if fitnessNew[i] < fitG:
            fitG = fitnessNew[i]
            Xgb[0, :] = positionNew[i, :]

    fs = ReliefF(n_neighbors=50)
    X_selected = fs.fit_transform(xtrain, ytrain)
    feature_scores = np.array(fs.feature_importances_)
    score_min = feature_scores.min()
    score_max = feature_scores.max()
    weight = (feature_scores - score_min) / (score_max - score_min)
    probability = A * np.sin(weight * math.pi) + B

    while t < max_iter:

        # fitness_sort_idx = np.argmin(fitnessNew)
        for i in range(N):
            if t == 0:
                w = 0.9 - 0.5 * (i / 1)
            else:
                w = 0.9 - 0.5 * (i / t)

        curve[0, t] = fitG.copy()
        # print("Iteration:", t + 1)
        # print("Best (PSO):", curve[0,t])
        t += 1
        # if self.gBestFit > self.fitnessNew[fitness_sort_idx]:
        #     self.gBest = copy.copy(self.positionNew[fitness_sort_idx, :])
        #     self.gBestFit = self.fitnessNew[fitness_sort_idx]
        #     self.gBestArr = copy.copy(self.pArrayNew[fitness_sort_idx, :])
        #     self.gBestAcc = self.accuracyNew[fitness_sort_idx]
        SPosition = copy.copy(positionNew)
        SVelocity = copy.copy(velocityNew)
        SFitness = copy.copy(fitnessNew)
        positionPre = copy.copy(positionNew)
        r1 = np.random.rand()
        r2 = np.random.rand()
        velocityNew = w * velocityPre + c1 * r1 * (pBest - positionPre) \
                           + c2 * r2 * (gBest - positionPre)
        velocityPre = velocityNew.copy()
        positionNew = positionPre + velocityNew
        positionNew = np.clip(positionNew, 0, 1)

        CPosition[:N, :] = copy.copy(SPosition)
        CVelocity[:N, :] = copy.copy(SVelocity)
        for Ni in range(Nc):
            for pop in range(N):
                for i in range(dim):
                    if np.random.rand() < probability[i]:
                        CPosition[pop + (Ni + 1) * N, i] = 1 - positionNew[pop, i]
                    CVelocity[pop + (Ni + 1) * N, :] = copy.copy(SVelocity[pop, :])
        distance_S = np.zeros((N * (Nc + 1), N))
        for i in range(N * (Nc + 1)):
            for j in range(N):
                distance_S[i, j] = distance(CPosition[i, :], SPosition[j, :])
        sortDistance_S = np.argsort(distance_S)
        for i in range(N * (Nc + 1)):
            CFitness[i] = np.mean(SFitness[sortDistance_S[:3]])

        CFitness1 = copy.copy(CFitness)
        CPosition1 = copy.copy(CPosition)
        CVelocity1 = copy.copy(CVelocity)
        position = np.zeros((N, dim))
        velocity = np.zeros((N, dim))
        for i in range(N):
            min_CFitness_idx = CFitness1.argmin()
            CFitness_min = CFitness1[min_CFitness_idx]
            CPosition_min = CPosition1[min_CFitness_idx, :]
            CVelocity_min = CVelocity1[min_CFitness_idx, :]
            CFitness1 = np.delete(CFitness1, min_CFitness_idx, axis=0)
            CPosition1 = np.delete(CPosition1, min_CFitness_idx, axis=0)
            CVelocity1 = np.delete(CVelocity1, min_CFitness_idx, axis=0)
            distance_arr = np.zeros(CFitness1.shape[0])
            for j in range(CFitness1.shape[0]):
                distance_arr[j] = distance(CPosition1[j, :], CPosition_min)
            nearst_idx = distance_arr.argmin()
            CFitness1 = np.delete(CFitness1, nearst_idx, axis=0)
            CPosition1 = np.delete(CPosition1, nearst_idx, axis=0)
            CVelocity1 = np.delete(CVelocity1, nearst_idx, axis=0)
            position[i, :] = CPosition_min
            velocity[i, :] = CVelocity_min


        positionNew = copy.copy(position)
        velocityNew = copy.copy(position)
        X_new = binary_conversion(positionNew, thres, N, dim)
        for i in range(N):
            fitnessNew[i] = Fun(xtrain, xvalid, ytrain, yvalid, X_new[i, :], opts)
            if fitnessNew[i] < pBestFit[i]:
                pBest[i, :] = copy.copy(positionNew[i, :])
                pBestFit[i] = fitnessNew[i]
            if fitnessNew[i] < fitG:
                fitG = fitnessNew[i]
                Xgb[0, :] = positionNew[i, :]

    # curves[0, :] = curve[0, :]
    # # 创建一个等比例的索引数组
    # indices = np.linspace(0, max_iter - 1, T)
    # 使用线性插值生成新的向量
    curves = np.zeros((1, T))
    o= 0

    for j in range(max_iter):
        for i in range(T // max_iter):
            curves[0, o] = curve[0, j]
            o += 1


    # Best feature subset
    Gbin = binary_conversion(Xgb, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    # Create dictionary
    pso_data = {'sf': Gbin, 'c': curves, 'nf': num_feat}
    return pso_data






