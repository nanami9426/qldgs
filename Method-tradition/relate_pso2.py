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


# 新增群体选择统计功能
# 新增精英统计类
class EliteTracker:
    def __init__(self, dim, elite_ratio=0.2):
        self.dim = dim
        self.elite_ratio = elite_ratio  # 精英比例(前20%)
        self.selection_counts = np.zeros(dim)
        self.total_elites = 0

    def update(self, fitness, masks):
        """更新精英选择统计
        masks: 当前所有粒子的特征掩码矩阵(N x dim)
        """
        # 选择表现最好的前elite_ratio比例粒子作为精英
        elite_num = max(1, int(len(fitness) * self.elite_ratio))
        elite_indices = np.argsort(fitness)[:elite_num]

        # 统计精英粒子选择的特征
        elite_masks = masks[elite_indices]
        self.selection_counts += np.sum(elite_masks, axis=0)
        self.total_elites += elite_num

    def get_elite_freq(self):
        """获取各特征的精英选择频率"""
        return self.selection_counts / (self.total_elites + 1e-6)


# 修改阈值更新函数
def adaptive_threshold_update(T, iteration, corr, best_num_feat, elite_tracker, target_dim=0):
    """
    精英引导的阈值更新策略

    参数变化：
    新增elite_tracker参数用于获取精英选择频率
    """
    # 基础增长率（时间相关）
    base_growth = 0.01 * np.log(iteration + 2)  # 对数增长更平缓

    # 获取精英选择频率
    elite_freq = elite_tracker.get_elite_freq()

    # 精英反馈因子（核心创新点）
    # 当精英选择频率>0.7时阈值下降，<0.3时加速上升
    elite_factor = np.where(
        elite_freq > 0.7,
        0.95 - 0.1 * (elite_freq - 0.7) / 0.3,  # 频率0.7→0.95, 1.0→0.85
        np.where(
            elite_freq < 0.3,
            1.05 + 0.15 * (0.3 - elite_freq) / 0.3,  # 频率0→1.2, 0.3→1.05
            1.0  # 中间区域保持中性
        )
    )

    # 相关性保护因子
    corr_factor = 0.6 + 0.4 * corr  # 高相关特征最大衰减到0.6倍

    # 目标维度调节
    target_factor = 1.0
    if target_dim > 0:
        error = (best_num_feat - target_dim) / target_dim
        target_factor = np.clip(1 + 0.15 * error, 0.85, 1.15)

    # 综合调整公式
    T_new = T * (1 + base_growth) * elite_factor * corr_factor * target_factor

    # 动态安全边界（基于相关性和精英频率）
    min_T = np.where(
        (corr > 0.6) | (elite_freq > 0.8),
        0.15,  # 高相关或高频精英特征最低阈值
        0.25
    )
    max_T = 0.35 + 0.6 * (1 - corr)  # 低相关特征允许更高阈值

    return np.clip(T_new, min_T, max_T)


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
    # 在初始化后添加统计器
    elite_tracker = EliteTracker(dim=dim, elite_ratio=0.2)
    # ...保持原有预处理不变...
    while t < max_iter:
        # 动态惯性权重
        w = 0.9
        # 二进制转换使用向量阈值
        Xbin = binary_conversion(X, T, N, dim)
        # tracker.update_stats(Xbin)  # 新增此行
        elite_tracker.update(fit[:, 0], Xbin)
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
        T = adaptive_threshold_update(T, t + 1, corr, current_num_feat, elite_tracker, target_dim)

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