#[2018]-"Parameter estimation of solar cells diode models by an improved opposition-based whale optimization algorithm"

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
    return np.clip(x, lb, ub)


#--- Opposition based learning (18)
def opposition_based_learning(X, lb, ub, thres, N, dim):
    Xo = lb + ub - X
                
    return Xo


def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # Parameters
    ub    = 1
    lb    = 0
    thres = 0.5
    b     = 1       # constant
    
    N        = opts['N']
    max_iter = opts['T']
    if 'b' in opts:
        b    = opts['b']
    
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X    = init_position(lb, ub, N, dim)
    
    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)
    
    # Fitness at first iteration
    fit  = np.zeros([N, 1], dtype='float')
    Xgb  = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
    
    for i in range(N):
        fit[i,0] = Fun(xtrain, xvalid, ytrain, yvalid, Xbin[i,:], opts)
        if fit[i,0] < fitG:
            Xgb[0,:] = X[i,:]
            fitG     = fit[i,0]

    #--- Opposition based learning
    Xo    = opposition_based_learning(X, lb, ub, thres, N, dim) 
    #--- Binary conversion
    Xobin = binary_conversion(Xo, thres, N, dim)
    
    #--- Fitness
    fitO  = np.zeros([N, 1], dtype='float')
    for i in range(N):
        fitO[i,0] = Fun(xtrain, xvalid, ytrain, yvalid, Xobin[i,:], opts)
        if fitO[i,0] < fitG:
            Xgb[0,:] = Xo[i,:]
            fitG     = fitO[i,0]
    
    #--- Merge opposite & current population, and select best N
    XX  = np.concatenate((X, Xo), axis=0)
    FF  = np.concatenate((fit, fitO), axis=0)
    #--- Sort in ascending order
    ind = np.argsort(FF, axis=0)
    for i in range(N):
        X[i,:]   = XX[ind[i,0],:]
        fit[i,0] = FF[ind[i,0]]
        
    # Pre
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    
    curve[0,t] = fitG.copy()
    # print("Generation:", t + 1)
    # print("Best (OBWOA):", curve[0,t])
    t += 1

    while t < max_iter:
        # Define a, linearly decreases from 2 to 0 (14)
        a = 2 - t * (2 / max_iter)
        
        for i in range(N):
            # Parameter A (13)
            A = 2 * a * rand() - a
            # Paramater C (13)
            C = 2 * rand()
            # Parameter r1, random number in [0,1]
            r1 = rand()
            # Parameter l, random number in [-1,1]
            l = -1 + 2 * rand()  
            # Whale position update (15)
            if r1  < 0.5:
                # {1} Encircling prey
                if abs(A) < 1:
                    for d in range(dim):
                        # Compute D (12)
                        Dx = abs(C * Xgb[0] - X[i])
                        # Position update (12)
                        X[i] = Xgb[0] - A * Dx
                        # Boundary
                        X[i] = boundary(X[i], lb[0], ub[0])
                
                # {2} Search for prey
                elif abs(A) >= 1:
                    # Select a random whale
                    k = np.random.randint(low=0, high=N)
                    # Compute D (16)
                    Dx = abs(C * X[k] - X[i])
                    # Position update (16)
                    X[i] = X[k] - A * Dx
                    # Boundary
                    X[i] = boundary(X[i], lb[0], ub[0])
            
            # {3} Bubble-net attacking 
            elif r1 >= 0.5:
                # Distance of whale to prey (11)
                dist = abs(X[i] - Xgb[0])
                # Position update (11)
                X[i] = dist * np.exp(b * l) * np.cos(2 * np.pi * l) + Xgb[0]
                # Boundary
                X[i] = boundary(X[i], lb[0], ub[0])
        
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, xvalid, ytrain, yvalid,  Xbin[i,:], opts)
            if fit[i,0] < fitG:
                Xgb[0,:] = X[i,:]
                fitG     = fit[i,0]

        #--- Opposition based learning
        Xo    = opposition_based_learning(X, lb, ub, thres, N, dim) 
        #--- Binary conversion
        Xobin = binary_conversion(Xo, thres, N, dim)
        
        #--- Fitness
        fitO  = np.zeros([N, 1], dtype='float')
        for i in range(N):
            fitO[i,0] = Fun(xtrain, xvalid, ytrain, yvalid,  Xobin[i,:], opts)
            if fitO[i,0] < fitG:
                Xgb[0,:] = Xo[i,:]
                fitG     = fitO[i,0]
        
        # Store result
        curve[0,t] = fitG.copy()
        # print("Generation:", t + 1)
        # print("Best (OBWOA):", curve[0,t])
        t += 1            

        #--- Merge opposite & current population, and select best N
        XX  = np.concatenate((X, Xo), axis=0)
        FF  = np.concatenate((fit, fitO), axis=0)
        #--- Sort in ascending order
        ind = np.argsort(FF, axis=0)
        for i in range(N):
            X[i,:]   = XX[ind[i,0],:]
            fit[i,0] = FF[ind[i,0]]
            
            
    # Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)    
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    obwoa_data = {'sf': Gbin, 'c': curve, 'nf': num_feat}
    
    return obwoa_data 
