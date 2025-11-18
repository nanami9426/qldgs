import numpy as np
from numpy.random import rand
from sklearn.feature_selection import mutual_info_classif  # 新增特征相关性计算
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
    """修改为支持向量阈值比较"""
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        # 向量化比较每个特征的独立阈值
        Xbin[i, :] = (X[i, :] > thres).astype(int)
    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    return x

def compute_feature_correlation(corr, rate):

    # 归一化到[0,1]并反转（高相关→低阈值）
    corr_norm = (corr - corr.min()) / (corr.max() - corr.min() + 1e-6)
    return (1 - corr_norm) * rate  # 阈值与相关性负相关


def adaptive_threshold_update(T, iteration, corr,best_num_feat, target_dim=0):
    """自适应阈值调整策略（阈值随时间递增版本）

    参数：
        T: 当前阈值向量
        iteration: 当前迭代次数
        best_num_feat: 当前最优特征数量
        target_dim: 目标特征维度

    返回：
        调整后的阈值向量
    """
    # 基础递增因子（每代最小增长幅度）
    base_growth = 1.01  # 1%的基础增长率

    # 动态调节参数
    error_ratio = 0.0
    if target_dim > 0:
        error_ratio = (best_num_feat - target_dim) / target_dim

    # 双重增长机制
    growth_factor = base_growth + 0.05 * np.tanh(2 * error_ratio)  # tanh限制调整幅度在±5%

    # 应用增长
    T_new = T * growth_factor

    # 安全边界约束（重要特征的阈值下限不同）
    # 根据初始相关性设置下限：高相关特征最低阈值不超过0.5
    # 假设已预先计算特征相关性corr（0-1范围，1表示高相关）
    min_T = np.where(corr > 0.7, 0.2, 0.4)  # 高相关特征最低阈值0.4，其他0.2
    max_T = 0.95

    # 应用边界
    T_new = np.clip(T_new, min_T, max_T)

    return T_new # 设置安全范围


def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # 参数解析
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

    # # Initialize position & velocity
    # X = init_position(lb, ub, N, dim)
    # V, Vmax, Vmin = init_velocity(lb, ub, N, dim)

    # Pre
    fit = np.zeros([N, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='float')
    XgbBin = np.zeros([1, dim], dtype='int')
    fitG = float('inf')
    Xpb = np.zeros([N, dim], dtype='float')
    fitP = float('inf') * np.ones([N, 1], dtype='float')
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0
    """计算特征相关性并生成初始阈值"""
    corr = mutual_info_classif(xtrain,ytrain)  # 互信息计算
    # 特征相关性初始化阈值
    T = compute_feature_correlation(corr, 0.7)
    # ...保持原有维度处理不变...
    # 初始化
    X = init_position(lb, ub, N, dim)
    V, Vmax, Vmin = init_velocity(lb, ub, N, dim)
    w_max = 0.9
    w_min = 0.2
    target_dim = 1
    T_best = T.copy()

    # ...保持原有预处理不变...
    while t < max_iter:
        # 动态惯性权重
        w = 0.9
        # 二进制转换使用向量阈值
        Xbin = binary_conversion(X, T, N, dim)
        for i in range(N):
            fit[i, 0] = Fun(xtrain, xvalid, ytrain, yvalid, Xbin[i, :], opts)
            if fit[i, 0] < fitP[i, 0]:
                Xpb[i, :] = X[i, :].copy()
                fitP[i, 0] = fit[i, 0]
            if fitP[i, 0] < fitG:
                Xgb[0, :] = Xpb[i, :].copy()
                fitG = fitP[i, 0]
                T_best = T.copy()
                # Store result
        curve[0, t] = fitG.copy()
        t += 1
        # ...保持适应度计算流程不变...

        # 在获得当前最优解后更新阈值
        current_num_feat = np.sum(binary_conversion(Xgb, T_best, 1, dim))
        T = adaptive_threshold_update(T, t + 1, corr,current_num_feat, target_dim)

        # ...保持粒子更新逻辑不变...
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
    Gbin = binary_conversion(Xgb, T_best, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    # Create dictionary
    pso_data = {'sf': Gbin, 'c': curve, 'nf': num_feat}

    # ...保持最终处理逻辑不变...

    return pso_data