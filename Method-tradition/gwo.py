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
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    alpha_rate = 0.9
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
    alpha_X = init_position(lb, ub, 1, dim)
    aXbin = binary_conversion( alpha_X, thres, 1, dim)
    afit = Fun(xtrain, xvalid, ytrain, yvalid, aXbin[0, :], opts)

    beta_X = init_position(lb, ub, 1, dim)
    bXbin = binary_conversion(beta_X, thres, 1, dim)
    bfit = Fun(xtrain, xvalid, ytrain, yvalid, bXbin[0, :], opts)
    delta_X = init_position(lb, ub, 1, dim)
    dXbin = binary_conversion(delta_X, thres, 1, dim)
    dfit = Fun(xtrain, xvalid, ytrain, yvalid, dXbin[0, :], opts)


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
            if fit[i, 0] < afit:
                afit = fit[i, 0]
                alpha_X[0, :] = X[i, :].copy()
            elif fit[i, 0] < bfit:
                bfit = fit[i, 0]
                beta_X[0, :] = X[i, :].copy()
            elif fit[i, 0] < dfit:
                dfit = fit[i, 0]
                delta_X[0, :] = X[i, :].copy()

        for i in range(N):
            newX = X[i, :].copy()
            for j in range(dim):
                r1, r2 = np.random.random(), np.random.random()
                A = 2 * alpha_rate * r1 - alpha_rate  # alpha调节因子
                C = 2 * r2  # alpha调节因子
                r3 =  np.random.random()
                D = 2 * r3

                # 更新特征选择
                newX[j] = X[i, j] + A * (alpha_X[0, j] - X[i, j]) + C * (beta_X[0, j] - X[i, j]) + D * (delta_X[0, j] - X[i, j])

                # 确保新的选择位置在[0, 1]范围内
                if newX[j] > 1:
                    newX[j] = 1
                if newX[j] < 0:
                    newX[j] = 0
            X[i, :] = newX.copy()





    # Best feature subset
    Gbin = binary_conversion(Xgb, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    # Create dictionary
    pso_data = {'sf': Gbin, 'c': curve, 'nf': num_feat}
    return pso_data