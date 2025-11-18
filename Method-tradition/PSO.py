import numpy as np
from numpy.random import rand
from Function import Fun

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
    # # Maximum & minimum velocity
    # for d in range(dim):
    #     Vmax[0, d] = (ub[0, d] - lb[0, d]) / 2
    #     Vmin[0, d] = -Vmax[0, d]
    # for i in range(N):
    #     for d in range(dim):
    #         V[i, d] = Vmin[0, d] + (Vmax[0, d] - Vmin[0, d]) * rand()
    # 计算Vmax和Vmin
    Vmax[0, :] = (ub[0, :] - lb[0, :]) / 2
    Vmin[0, :] = -Vmax[0, :]

    # 生成随机值并更新V
    V = Vmin + (Vmax - Vmin) * np.random.rand(N, dim)
    return V, Vmax, Vmin

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        Xbin[i, :] = (X[i, :] > thres).astype(int)
    return Xbin

def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    return x

def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    w = 0.9  # inertia weight
    c1 = 2  # acceleration factor
    c2 = 2  # acceleration factor
    N = opts['N']
    max_iter = opts['T']
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

    # Initialize position & velocity
    X = init_position(lb, ub, N, dim)
    V, Vmax, Vmin = init_velocity(lb, ub, N, dim)

    # Pre
    fit = np.zeros([N, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
    Xpb = np.zeros([N, dim], dtype='float')
    fitP = float('inf') * np.ones([N, 1], dtype='float')
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    while t < max_iter:
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        # Fitness
        for i in range(N):
            fit[i, 0] = Fun(xtrain, xvalid, ytrain, yvalid, Xbin[i, :], opts)
            if fit[i, 0] < fitP[i, 0]:
                Xpb[i, :] = X[i, :]
                fitP[i, 0] = fit[i, 0]
            if fitP[i, 0] < fitG:
                Xgb[0, :] = Xpb[i, :]
                fitG = fitP[i, 0]
        # Store result
        curve[0, t] = fitG.copy()
        # print("Iteration:", t + 1)
        # print("Best (PSO):", curve[0,t])
        t += 1
        for i in range(N):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            # Update velocity for all dimensions
            V[i, :] = w * V[i, :] + c1 * r1 * (Xpb[i, :] - X[i, :]) + c2 * r2 * (Xgb[0, :] - X[i, :])

            # Boundary check for velocity
            V[i, :] = np.clip(V[i, :], Vmin[0, :], Vmax[0, :])

            # Update position for all dimensions
            X[i, :] = X[i, :] + V[i, :]

            # Boundary check for position
            X[i, :] = np.clip(X[i, :], lb[0, :], ub[0, :])

    # Best feature subset
    Gbin = binary_conversion(Xgb, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    # Create dictionary
    pso_data = {'sf': Gbin, 'c': curve, 'nf': num_feat}
    return pso_data