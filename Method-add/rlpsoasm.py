import numpy as np
from math import log2, sqrt

# -----------------------------
# 通用工具：离散标签 & 熵
# -----------------------------
def encode_labels(labels):
    unique, inv = np.unique(labels, return_inverse=True)
    return inv, unique

def entropy_from_counts(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return -np.sum(p * np.log2(p))


# -----------------------------
# SU 计算（式 (7)）
# -----------------------------
def symmetrical_uncertainty(X, y, num_bins=10, random_state=None):
    """
    逐特征计算 Symmetrical Uncertainty SU(X_i, y)
    使用基于秩的分箱保证近似等频分箱。
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, float)
    y_enc, _ = encode_labels(y)
    n_samples, n_features = X.shape

    # H(y)
    counts_y = np.bincount(y_enc)
    Hy = entropy_from_counts(counts_y)

    su_scores = np.zeros(n_features)
    for j in range(n_features):
        col = X[:, j]
        # 用秩作等频分箱：0,...,num_bins-1
        order = np.argsort(col)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(n_samples)
        bins = (ranks * num_bins) // n_samples

        max_bin = num_bins
        max_y = y_enc.max() + 1
        joint = np.zeros((max_bin, max_y), dtype=int)
        for b, cls in zip(bins, y_enc):
            joint[b, cls] += 1

        counts_x = joint.sum(axis=1)
        Hx = entropy_from_counts(counts_x)

        joint_flat = joint.flatten()
        Hxy = entropy_from_counts(joint_flat)

        Ixy = Hx + Hy - Hxy
        if Hx + Hy == 0:
            su = 0.0
        else:
            su = 2 * Ixy / (Hx + Hy)
        su_scores[j] = su

    return su_scores


# -----------------------------
# ReliefF（近似实现，对应 ASM 中第二个打分器）
# -----------------------------
def relieff_scores(X, y, n_neighbors=5, n_samples=100, random_state=None):
    """
    数值型 ReliefF，采样 n_samples 个样本近似权重。
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, float)
    y_enc, classes = encode_labels(y)
    n_samples_total, n_features = X.shape
    n_classes = len(classes)
    n_samples = min(n_samples, n_samples_total)

    # 先验概率 P(C)
    counts = np.bincount(y_enc)
    priors = counts / counts.sum()

    # 特征范围用于数值 diff 归一化
    f_min = X.min(axis=0)
    f_max = X.max(axis=0)
    f_range = f_max - f_min
    f_range[f_range == 0] = 1.0

    W = np.zeros(n_features)

    sample_indices = rng.choice(n_samples_total, size=n_samples, replace=False)
    for idx in sample_indices:
        x_r = X[idx]
        y_r = y_enc[idx]
        diff = X - x_r
        dists = np.sqrt(np.sum(diff * diff, axis=1))

        for c in range(n_classes):
            mask_c = (y_enc == c)
            if c == y_r:
                mask_c[idx] = False
            idx_c = np.where(mask_c)[0]
            if len(idx_c) == 0:
                continue
            dists_c = dists[idx_c]
            k = min(n_neighbors, len(idx_c))
            nn_idx_local = np.argpartition(dists_c, k-1)[:k]
            nn_idx = idx_c[nn_idx_local]

            diff_feat = np.abs(X[nn_idx] - x_r) / f_range
            mean_diff = diff_feat.mean(axis=0)
            if c == y_r:
                # hits
                W -= mean_diff / n_samples
            else:
                # misses
                W += (priors[c] / (1 - priors[y_r])) * mean_diff / n_samples

    return W


# -----------------------------
# ASM：SU + ReliefF + 距离阈值 d_t
# -----------------------------
def adaptive_scoring_mechanism(X, y, dt=0.5, num_bins=10,
                               relieff_neighbors=5, relieff_samples=100,
                               random_state=None):
    """
    ASM 两阶段（这里按论文描述做了一个合理重构）：
      1) 计算 SU 与 ReliefF 打分；
      2) 标准化后看 (SU, ReliefF) 在 2D 空间中的欧氏距离；
         距离 >= d_t 的特征视为“高质量特征”。
    返回：
      F_idx: 选中的高质量特征下标
      scores: { 'su', 'relieff', 'dist' }
    """
    su = symmetrical_uncertainty(X, y, num_bins=num_bins, random_state=random_state)
    rel = relieff_scores(X, y,
                         n_neighbors=relieff_neighbors,
                         n_samples=relieff_samples,
                         random_state=random_state)

    def norm01(v):
        v = np.asarray(v, float)
        vmin = v.min()
        vmax = v.max()
        if vmax - vmin == 0:
            return np.zeros_like(v)
        return (v - vmin) / (vmax - vmin)

    su_n = norm01(su)
    rel_n = norm01(rel)
    dist = np.sqrt(su_n**2 + rel_n**2)

    F_idx = np.where(dist >= dt)[0]
    if F_idx.size == 0:
        # 如果 d_t 太严，至少保留前 10%
        D = X.shape[1]
        k = max(1, int(0.1 * D))
        F_idx = np.argsort(dist)[-k:]

    scores = {'su': su, 'relieff': rel, 'dist': dist}
    return F_idx, scores


# -----------------------------
# 二值向量 NMI（用于 EC-HIM 中的相似度）
# -----------------------------
def nmi_binary(a, b):
    """
    两个 0/1 向量的 NMI，形式类似 SU：2I / (H(X)+H(Y))
    """
    a = np.asarray(a).astype(int).ravel()
    b = np.asarray(b).astype(int).ravel()
    assert a.shape == b.shape
    n = a.size

    n11 = np.sum((a == 1) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0))
    n01 = np.sum((a == 0) & (b == 1))
    n00 = n - n11 - n10 - n01

    counts = np.array([[n00, n01],
                       [n10, n11]], dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        num = p
        den = px @ py
        ratio = np.where(num > 0, num / den, 1.0)
        I = np.sum(np.where(num > 0, num * np.log2(ratio), 0.0))

    Hx = entropy_from_counts(np.array([n00 + n01, n10 + n11]))
    Hy = entropy_from_counts(np.array([n00 + n10, n01 + n11]))
    if Hx + Hy == 0:
        return 0.0
    return 2 * I / (Hx + Hy)


# -----------------------------
# Logistic 混沌序列（EC-HIM）
# -----------------------------
def logistic_population(pop_size, dim, mu=4.0, random_state=None):
    rng = np.random.default_rng(random_state)
    Y = np.zeros((pop_size, dim))
    Y[0] = rng.random(dim)
    for i in range(1, pop_size):
        Y[i] = mu * Y[i-1] * (1.0 - Y[i-1])
    return Y


# -----------------------------
# KNN-1 + 10 折交叉验证误差
# -----------------------------
def knn1_cv_error(X, y, n_folds=10, random_state=None):
    """
    1-NN, 欧式距离, n_folds 折交叉验证，返回错误率。
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, float)
    y = np.asarray(y)
    n_samples = X.shape[0]

    indices = rng.permutation(n_samples)
    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[: n_samples % n_folds] += 1
    current = 0
    errors = 0
    total = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        current = stop

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        for x, true_label in zip(X_val, y_val):
            diff = X_train - x
            dists = np.sqrt(np.sum(diff * diff, axis=1))
            nn_idx = np.argmin(dists)
            pred = y_train[nn_idx]
            if pred != true_label:
                errors += 1
            total += 1
    return errors / total if total > 0 else 0.0


# -----------------------------
# 适应度函数：式 (21)
# -----------------------------
def fitness_fs(position_continuous, X, y, selected_feature_indices,
               fs_threshold=0.6, beta=0.9, random_state=None):
    """
    按式 (21) 计算 FS 适应度：
      fitness = beta * Error + (1-beta) * (d / D)
    其中 Error 为 KNN(k=1) 10 折 CV 的错误率。
    """
    pos = np.asarray(position_continuous)
    F_idx = np.asarray(selected_feature_indices)
    assert pos.shape[0] == F_idx.shape[0]

    # 子空间二值掩码
    mask_sub = pos >= fs_threshold
    if not np.any(mask_sub):
        # 如果全 0，则强制至少选一维：取位置值最大的
        best_idx_sub = np.argmax(pos)
        mask_sub[best_idx_sub] = True

    D = X.shape[1]
    mask_full = np.zeros(D, dtype=bool)
    mask_full[F_idx[mask_sub]] = True
    d = mask_full.sum()

    X_sub = X[:, mask_full]
    error = knn1_cv_error(X_sub, y, n_folds=10, random_state=random_state)
    fitness = beta * error + (1 - beta) * (d / D)
    return fitness, mask_full, mask_sub


# -----------------------------
# EC-HIM 初始化（算法 2 重构）
# -----------------------------
def ec_him_initialization(X, y, F_idx, pop_size=50, fs_threshold=0.6,
                          beta=0.9, mu=4.0, random_state=None):
    """
    EC-HIM:
      - 随机 + Logistic 混沌产生 2N 个个体
      - 以适应度排序划分精英/普通
      - 利用 NMI 找到与精英差异最大的普通个体，进行均匀交叉
      - 择优保留精英 or 子代
    """
    rng = np.random.default_rng(random_state)
    d_sub = len(F_idx)

    pop_rand = rng.random((pop_size, d_sub))
    pop_chaos = logistic_population(pop_size, d_sub, mu=mu, random_state=rng)
    pop_all = np.vstack([pop_rand, pop_chaos])
    n_all = pop_all.shape[0]

    fitness_vals = np.zeros(n_all)
    for i in range(n_all):
        f, _, _ = fitness_fs(pop_all[i], X, y, F_idx,
                             fs_threshold=fs_threshold, beta=beta,
                             random_state=random_state)
        fitness_vals[i] = f

    idx_sorted = np.argsort(fitness_vals)
    elite_idx = idx_sorted[:pop_size]
    common_idx = idx_sorted[pop_size:]

    elite_pop = pop_all[elite_idx].copy()
    common_pop = pop_all[common_idx].copy()
    elite_fit = fitness_vals[elite_idx].copy()

    new_pop = np.zeros((pop_size, d_sub))
    elite_bin = elite_pop >= fs_threshold
    common_bin = common_pop >= fs_threshold

    for i in range(pop_size):
        x_e = elite_pop[i]
        bin_e = elite_bin[i]

        nmies = np.array([nmi_binary(bin_e, cb) for cb in common_bin])
        j = np.argmin(nmies)
        x_c = common_pop[j]

        mask = rng.random(d_sub) < 0.5
        child1 = np.where(mask, x_e, x_c)
        child2 = np.where(mask, x_c, x_e)
        f1, _, _ = fitness_fs(child1, X, y, F_idx,
                              fs_threshold=fs_threshold, beta=beta,
                              random_state=random_state)
        f2, _, _ = fitness_fs(child2, X, y, F_idx,
                              fs_threshold=fs_threshold, beta=beta,
                              random_state=random_state)
        if f1 < f2:
            x_i, fit_i = child1, f1
        else:
            x_i, fit_i = child2, f2

        if elite_fit[i] < fit_i:
            new_pop[i] = x_e
        else:
            new_pop[i] = x_i

        common_pop = np.delete(common_pop, j, axis=0)
        common_bin = np.delete(common_bin, j, axis=0)

    return new_pop


# -----------------------------
# 核心 RLPSO-ASM 循环（算法 3 + 4）
# -----------------------------
def rl_pso_asm_fs(X, y,
                   pop_size=50,
                   max_iter=100,
                   beta=0.9,
                   c1=1.49445,
                   c2=1.49445,
                   w_max=0.9,
                   w_min=0.4,
                   mu=4.0,
                   rg=0.2,
                   alpha_guiding=10.0,
                   dt=0.5,
                   fs_threshold=0.6,
                   relieff_neighbors=5,
                   relieff_samples=100,
                   num_bins=10,
                   q_alpha=0.1,
                   q_gamma=0.9,
                   q_temperature=1.0,
                   random_state=None):
    """
    RLPSO-ASM 主算法。返回：
      Gbin_full: (D,) 0/1 选择掩码
      curve:     长度为 max_iter 的最优适应度曲线
      num_feat:  选中特征数
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, float)
    y = np.asarray(y)
    n_samples, D = X.shape

    # 1. ASM：找高质量特征子空间 F_idx
    F_idx, asm_scores = adaptive_scoring_mechanism(
        X, y, dt=dt,
        num_bins=num_bins,
        relieff_neighbors=relieff_neighbors,
        relieff_samples=relieff_samples,
        random_state=random_state
    )
    d_sub = len(F_idx)
    if d_sub == 0:
        F_idx = np.arange(D)
        d_sub = D

    # 2. EC-HIM 初始化
    positions = ec_him_initialization(
        X, y, F_idx,
        pop_size=pop_size,
        fs_threshold=fs_threshold,
        beta=beta,
        mu=mu,
        random_state=random_state
    )
    velocities = np.zeros_like(positions)

    pop_size_eff = positions.shape[0]
    fitness_vals = np.zeros(pop_size_eff)
    gbest_fit = np.inf
    gbest_pos_sub = None
    gbest_mask_full = None

    for i in range(pop_size_eff):
        f, mask_full, _ = fitness_fs(
            positions[i], X, y, F_idx,
            fs_threshold=fs_threshold,
            beta=beta,
            random_state=random_state
        )
        fitness_vals[i] = f
        if f < gbest_fit:
            gbest_fit = f
            gbest_pos_sub = positions[i].copy()
            gbest_mask_full = mask_full.copy()

    # 3. RL 驱动的学习框架 + PSO 更新
    Q = np.zeros((2, 3))  # 状态 S1/S2 × 动作 (三种策略)
    curve = []

    for t in range(1, max_iter + 1):
        # ω = 0.9 - 0.5 * (t/T)
        w = w_max - (w_max - w_min) * (t / max_iter)

        # 引导子群大小 n_g（式 (13)）
        ng = rg * pop_size_eff + alpha_guiding * (1 - t / max_iter)
        ng_int = int(np.clip(round(ng), 1, pop_size_eff))

        idx_sorted = np.argsort(fitness_vals)
        guiding_idx = idx_sorted[:ng_int]
        discovery_idx = idx_sorted[ng_int:]
        fit_guiding_threshold = fitness_vals[guiding_idx[-1]]

        # 邻域规模 k（式 (18)）
        k_nn = max(3, int(round(10 * (1 - t / max_iter))))
        k_nn = min(k_nn, pop_size_eff - 1) if pop_size_eff > 1 else 0

        # 精英池规模 N_t（式 (16)）
        if t < 0.3 * max_iter:
            elite_count = int(round(0.3 * pop_size_eff))
        elif t < 0.7 * max_iter:
            elite_count = int(round(0.2 * pop_size_eff))
        else:
            elite_count = int(round(0.1 * pop_size_eff))
        elite_count = max(1, min(elite_count, pop_size_eff))
        elite_idx = idx_sorted[:elite_count]

        state_S1_mask = np.zeros(pop_size_eff, dtype=bool)
        state_S1_mask[guiding_idx] = True

        new_positions = np.zeros_like(positions)
        new_velocities = np.zeros_like(velocities)
        new_fitness = np.zeros_like(fitness_vals)

        for i in range(pop_size_eff):
            pos_i = positions[i]
            vel_i = velocities[i]
            f_i = fitness_vals[i]

            s = 0 if state_S1_mask[i] else 1  # S1 / S2

            # Softmax 选动作（式 (5)）
            q_row = Q[s]
            q_shift = q_row - np.max(q_row)
            exps = np.exp(q_shift / q_temperature)
            if np.all(exps == 0):
                probs = np.full_like(exps, 1.0 / len(exps))
            else:
                probs = exps / exps.sum()
            a = rng.choice(3, p=probs)

            # 三种动作对应三种学习策略（式 (14), (15), (17)）
            if a == 0:  # multi-guided
                if guiding_idx.size == 0:
                    pg_idx = i
                else:
                    pg_idx = rng.choice(guiding_idx)
                if discovery_idx.size == 0:
                    pd_idx = rng.integers(pop_size_eff)
                else:
                    pd_idx = rng.choice(discovery_idx)
                pg = positions[pg_idx]
                pd = positions[pd_idx]
                r1 = rng.random(d_sub)
                r2 = rng.random(d_sub)
                vel_new = (w * vel_i +
                           c1 * r1 * (pg - pos_i) +
                           c2 * r2 * (pd - pos_i))
                pos_new = pos_i + vel_new

            elif a == 1:  # top-guided
                li_idx = rng.choice(elite_idx)
                li = positions[li_idx]
                gbest = gbest_pos_sub if gbest_pos_sub is not None else pos_i
                r1 = rng.random(d_sub)
                r2 = rng.random(d_sub)
                vel_new = (w * vel_i +
                           c1 * r1 * (li - pos_i) +
                           c2 * r2 * (gbest - pos_i))
                pos_new = pos_i + vel_new

            else:  # neighbor-guided
                if pop_size_eff > 1 and k_nn > 0:
                    diff = positions - pos_i
                    dists = np.sqrt(np.sum(diff * diff, axis=1))
                    dists[i] = np.inf
                    k_eff = min(k_nn, pop_size_eff - 1)
                    nn_idx = np.argpartition(dists, k_eff)[:k_eff]
                    d_nn = dists[nn_idx]
                    d_nn = np.where(d_nn == 0, 1e-12, d_nn)
                    wj = 1.0 / d_nn
                    wj /= wj.sum()
                    x_center = np.sum(positions[nn_idx] * wj[:, None], axis=0)
                else:
                    x_center = pos_i.copy()
                r1 = rng.random(d_sub)
                vel_new = w * vel_i + c1 * r1 * (x_center - pos_i)
                pos_new = pos_i + vel_new

            pos_new = np.clip(pos_new, 0.0, 1.0)
            vel_new = pos_new - pos_i

            f_new, mask_full_new, _ = fitness_fs(
                pos_new, X, y, F_idx,
                fs_threshold=fs_threshold,
                beta=beta,
                random_state=random_state
            )

            if f_new < f_i:
                r = 1.0
                new_positions[i] = pos_new
                new_velocities[i] = vel_new
                new_fitness[i] = f_new
            else:
                r = 0.0
                new_positions[i] = pos_i
                new_velocities[i] = vel_i
                new_fitness[i] = f_i

            # 下一状态估计
            s_next = 0 if new_fitness[i] <= fit_guiding_threshold else 1

            # Q-learning 更新（式 (6) 形式）
            Q[s, a] = ((1 - q_alpha) * Q[s, a] +
                       q_alpha * (r + q_gamma * np.max(Q[s_next])))

            if new_fitness[i] < gbest_fit:
                gbest_fit = new_fitness[i]
                gbest_pos_sub = new_positions[i].copy()
                gbest_mask_full = mask_full_new.copy()

        positions = new_positions
        velocities = new_velocities
        fitness_vals = new_fitness
        curve.append(float(gbest_fit))

    if gbest_mask_full is None:
        Gbin_full = np.zeros(D, dtype=int)
        Gbin_full[F_idx] = 1
    else:
        Gbin_full = gbest_mask_full.astype(int)

    num_feat = int(Gbin_full.sum())
    return Gbin_full, curve, num_feat


# -----------------------------
# 对外接口 fs()
# -----------------------------
def fs(xtrain, xvalid, ytrain, yvalid, opts=None):
    """
    复现 RLPSO-ASM 的特征选择函数。
    输入:
      xtrain: (n_train, n_features) 训练集特征
      xvalid: (n_valid, n_features) 验证集特征
      ytrain: (n_train,) 训练标签
      yvalid: (n_valid,) 验证标签
      opts:   可选参数字典，默认完全按论文设置（可覆盖）

    输出:
      {
        'sf': Gbin,   # 0/1 特征选择向量（长度为原始维度 D）
        'c' : curve,  # 每次迭代的全局最优适应度
        'nf': num_feat  # 被选中特征个数
      }
    """
    if opts is None:
        opts = {}

    # RLPSO-ASM 本身基于内部 10 折交叉验证，因此拼接训练/验证数据保持算法思想不变
    X = np.concatenate((xtrain, xvalid), axis=0)
    y = np.concatenate((ytrain, yvalid), axis=0)

    pop_size = opts.get('N', 50)
    max_iter = opts.get('T', 100)
    beta = opts.get('beta', 0.9)
    c1 = opts.get('c1', 1.49445)
    c2 = opts.get('c2', 1.49445)
    w_max = opts.get('w_max', 0.9)
    w_min = opts.get('w_min', 0.4)
    mu = opts.get('mu', 4.0)
    rg = opts.get('rg', 0.2)
    alpha_guiding = opts.get('alpha_guiding', 10.0)
    dt = opts.get('dt', 0.5)
    fs_threshold = opts.get('fs_threshold', 0.6)
    relieff_neighbors = opts.get('relieff_neighbors', 5)
    relieff_samples = opts.get('relieff_samples', 100)
    num_bins = opts.get('num_bins', 10)
    q_alpha = opts.get('q_alpha', 0.1)
    q_gamma = opts.get('q_gamma', 0.9)
    q_temperature = opts.get('q_temperature', 1.0)
    random_state = opts.get('random_state', opts.get('random_seed', None))

    Gbin, curve, num_feat = rl_pso_asm_fs(
        X, y,
        pop_size=pop_size,
        max_iter=max_iter,
        beta=beta,
        c1=c1,
        c2=c2,
        w_max=w_max,
        w_min=w_min,
        mu=mu,
        rg=rg,
        alpha_guiding=alpha_guiding,
        dt=dt,
        fs_threshold=fs_threshold,
        relieff_neighbors=relieff_neighbors,
        relieff_samples=relieff_samples,
        num_bins=num_bins,
        q_alpha=q_alpha,
        q_gamma=q_gamma,
        q_temperature=q_temperature,
        random_state=random_state
    )

    return {'sf': Gbin, 'c': curve, 'nf': num_feat}
