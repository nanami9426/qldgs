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

def update_position(bacteria, i):
    """更新细菌的位置"""
    new_bacteria = bacteria[i].copy()
    move_direction = np.random.choice([1, -1])
    move_feature = np.random.randint(0, n_features - 1)

    # 根据步长和方向更新位置
    new_bacteria[move_feature] = (new_bacteria[move_feature] + move_direction) % 2  # Flip the feature state
    selected_features = np.where(new_bacteria == 1)[0]  # 更新后选中的特征

    if len(selected_features) > max_selected_features:  # 限制选特征数目
        return bacteria[i]  # 保持原位置

    return new_bacteria

 # 初始化细菌位置
def initialize_bacteria():
    bacteria = []
    for _ in range(n_bacteria):
        # 随机生成细菌的位置（特征子集）
        solution = np.random.randint(2, size=n_features)  # 二进制编码
        selected_features = np.sum(solution)
        if selected_features > max_selected_features:  # 确保选择的特征数不超过最大特征数
            solution = initialize_bacteria()  # 重新初始化
        bacteria.appen

def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    w = 0.9  # inertia weight
    step_size = 0.1
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


    X = np.random.randint(2, size=(N,dim))  # 二进制编码
    fit = np.zeros([N, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
    Xpb = np.zeros([N, dim], dtype='float')
    fitP = float('inf') * np.ones([N, 1], dtype='float')
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0
    while t < max_iter:
       Xbin = binary_conversion(X, thres, N, dim)
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
           new_X = X[i].copy()
           move_direction = np.random.choice([1, -1])
           move_feature = np.random.randint(0, dim - 1)

           # 根据步长和方向更新位置
           X[move_feature] = (new_X[move_feature] + move_direction) % 2  # Flip the feature state
           # selected_features = np.where(new_X == 1)[0]  # 更新后选中的特征

           # Best feature subset
    Gbin = binary_conversion(Xgb, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    # Create dictionary
    bco_data = {'sf': Gbin, 'c': curve, 'nf': num_feat}
    return bco_data






