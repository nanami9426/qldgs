#[2003]-"Bare bones particle swarms"

import numpy as np
from numpy.random import rand
from Function import Fun


def init_position(lb, ub, N, dim):
    X = lb + (ub - lb) * np.random.rand(N, dim)
    
    return X


def binary_conversion(X, thres, N, dim):
    Xbin = np.where(X > thres, 1, 0)
    
    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x
    

def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # Parameters
    ub    = 1
    lb    = 0
    thres = 0.5
    
    N        = opts['N']
    max_iter = opts['T']
    
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X     = init_position(lb, ub, N, dim)
    
    # Pre
    fit   = np.zeros([N, 1], dtype='float')
    Xgb   = np.zeros([1, dim], dtype='float')
    fitG  = float('inf')
    Xpb   = np.zeros([N, dim], dtype='float')
    fitP  = float('inf') * np.ones([N, 1], dtype='float')
    curve = np.zeros([1, max_iter], dtype='float') 
    t = 0
    
    while t < max_iter:
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, xvalid, ytrain, yvalid, Xbin[i,:], opts)
            if fit[i,0] < fitP[i,0]:
                Xpb[i,:]  = X[i,:]
                fitP[i,0] = fit[i,0]
                
            if fitP[i,0] < fitG:
                Xgb[0,:]  = Xpb[i,:]
                fitG      = fitP[i,0]
        
        # Store result
        curve[0,t] = fitG.copy()
        # print("Iteration:", t + 1)
        # print("Best (BBPSO):", curve[0,t])
        t += 1

        # --- Mean
        mu = (Xpb + Xgb[0]) / 2  # Shape: (N, dim)
        # --- Standard deviation
        sd = np.abs(Xpb - Xgb[0])  # Shape: (N, dim)
        # --- Gaussian random number
        X = sd * np.random.randn(N, dim) + mu  # Shape: (N, dim)
    
                
    # Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    # bbpso_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    bbpso_data = {'sf': Gbin, 'c': curve, 'nf': num_feat}
    
    return bbpso_data    
    
    







