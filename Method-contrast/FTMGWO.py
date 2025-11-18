#[2020]-"A new fusion of grey wolf optimizer algorithm with a two-phase mutation for feature selection"

import numpy as np
from numpy.random import rand
from Function import Fun


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
    return X


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x
 

#--- transfer function update binary position (4.3.2)
def transfer_function(x):
    Xs = abs(np.tanh(x))

    return Xs


def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # Parameters
    ub    = 1
    lb    = 0
    thres = 0.5

    
    N        = opts['N']
    max_iter = opts['T']
    if 'Mp' in opts:
        Mp   = opts['Mp']   
        
    # Dimension
    dim = np.size(xtrain, 1)
    if dim < 100:
        Mp = 0.5
    elif dim < 1000:
        Mp = 0.3
    else:
        Mp = 0.05  # mutation probability
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X  = init_position(lb, ub, N, dim)
    
    #--- Binary conversion
    X  = binary_conversion(X, thres, N, dim)
    
    # Fitness at first iteration
    fit    = np.zeros([N, 1], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='int')
    Xbeta  = np.zeros([1, dim], dtype='int')
    Xdelta = np.zeros([1, dim], dtype='int')
    Falpha = float('inf')
    Fbeta  = float('inf')
    Fdelta = float('inf')
    
    for i in range(N):
        fit[i,0] = Fun(xtrain, xvalid, ytrain, yvalid, X[i,:], opts)
        if fit[i,0] < Falpha:
            Xalpha[0,:] = X[i,:]
            Falpha      = fit[i,0]
        if fit[i,0] < Fbeta and fit[i,0] > Falpha:
            Xbeta[0,:]  = X[i,:]
            Fbeta       = fit[i,0]
        if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
            Xdelta[0,:] = X[i,:]
            Fdelta      = fit[i,0]
    
    # Pre
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    
    curve[0,t] = Falpha.copy()
    # print("Iteration:", t + 1)
    # print("Best (TMGWO):", curve[0,t])
    t += 1
    
    while t < max_iter:  
      	# Coefficient decreases linearly from 2 to 0 (3.5)
        a = 2 - t * (2 / max_iter)
        for i in range(N):
            # 生成随机参数 C1, C2, C3
            C1, C2, C3 = 2 * np.random.rand(dim), 2 * np.random.rand(dim), 2 * np.random.rand(dim)

            # 计算 Dalpha, Dbeta, Ddelta (3.7 - 3.9)
            Dalpha = np.abs(C1 * Xalpha - X[i])
            Dbeta = np.abs(C2 * Xbeta - X[i])
            Ddelta = np.abs(C3 * Xdelta - X[i])

            # 计算 A1, A2, A3 (3.3)
            A1, A2, A3 = 2 * a * np.random.rand(dim) - a, 2 * a * np.random.rand(dim) - a, 2 * a * np.random.rand(
                dim) - a

            # 计算 X1, X2, X3 (3.7 - 3.9)
            X1 = Xalpha - A1 * Dalpha
            X2 = Xbeta - A2 * Dbeta
            X3 = Xdelta - A3 * Ddelta

            # 计算新的位置 Xn (3.6)
            Xn = (X1 + X2 + X3) / 3

            # 计算二进制转换概率 (transfer function)
            Xs = transfer_function(Xn)

            # 生成随机数，并更新位置 (4.3.2)
            X[i] = np.where(np.random.rand(dim) < Xs, 1, 0)
        
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, xvalid, ytrain, yvalid, X[i,:], opts)
            if fit[i,0] < Falpha:
                Xalpha[0,:] = X[i,:]
                Falpha = fit[i,0]
            if fit[i,0] < Fbeta and fit[i,0] > Falpha:
                Xbeta[0,:]  = X[i,:]
                Fbeta = fit[i,0]
            if fit[i,0] < Fdelta and fit[i,0] > Fbeta and fit[i,0] > Falpha:
                Xdelta[0,:] = X[i,:]
                Fdelta      = fit[i,0]
        
        curve[0,t] = Falpha.copy()
        # print("Iteration:", t + 1)
        # print("Best (TMGWO):", curve[0,t])
        t += 1

        # --- two phase mutation: first phase
        # find index of 1
        idx1 = np.where(Xalpha == 1)[1]
        r1 = np.random.rand(len(idx1))  # Generate random numbers for all indices of 1
        mutate_indices1 = idx1[r1 < Mp]  # Find indices to mutate
        Xmut1 = Xalpha.copy()  # Copy the original array
        Xmut1[0, mutate_indices1] = 0  # Mutate selected indices to 0

        # Evaluate the mutated solution
        Fnew1 = Fun(xtrain, xvalid, ytrain, yvalid, Xmut1[0, :], opts)
        if Fnew1 < Falpha:
            Falpha = Fnew1
            Xalpha[0, :] = Xmut1[0, :]

        # --- two phase mutation: second phase
        # find index of 0
        idx0 = np.where(Xalpha == 0)[1]
        r2 = np.random.rand(len(idx0))  # Generate random numbers for all indices of 0
        mutate_indices2 = idx0[r2 < Mp]  # Find indices to mutate
        Xmut2 = Xalpha.copy()  # Copy the original array
        Xmut2[0, mutate_indices2] = 1  # Mutate selected indices to 1

        # Evaluate the mutated solution
        Fnew2 = Fun(xtrain, xvalid, ytrain, yvalid, Xmut2[0, :], opts)
        if Fnew2 < Falpha:
            Falpha = Fnew2
            Xalpha[0, :] = Xmut2[0, :]
                
        
    # Best feature subset
    Gbin       = Xalpha[0,:]
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    tmgwo_data = {'sf': Gbin, 'c': curve, 'nf': num_feat}
    
    return tmgwo_data    
                
                
                
    
