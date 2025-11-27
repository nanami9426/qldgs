import numpy as np

# ------------------------------------------------------------
# 基础工具：KMeans（无依赖版）、one-hot、kNN 图、拉普拉斯
# ------------------------------------------------------------

def kmeans(X, n_clusters, n_init=10, max_iter=100, random_state=None):
    """
    简单 KMeans 实现，X: (n_samples, n_features)
    返回 labels: (n_samples,)
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    if n_samples < n_clusters:
        raise ValueError("n_samples < n_clusters")

    best_inertia = None
    best_labels = None

    for _ in range(n_init):
        # 随机初始化簇中心
        init_idx = rng.choice(n_samples, size=n_clusters, replace=False)
        centroids = X[init_idx].copy()

        for _ in range(max_iter):
            # 计算到簇中心的距离并分配标签
            diff = X[:, None, :] - centroids[None, :, :]   # (n, k, d)
            dist2 = np.sum(diff**2, axis=2)                # (n, k)
            labels = np.argmin(dist2, axis=1)

            # 更新簇中心
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                pts = X[labels == k]
                if len(pts) == 0:
                    new_centroids[k] = X[rng.randint(0, n_samples)]
                else:
                    new_centroids[k] = pts.mean(axis=0)

            if np.allclose(centroids, new_centroids):
                centroids = new_centroids
                break
            centroids = new_centroids

        # 计算当前初始化的 inertia
        diff = X - centroids[labels]
        inertia = np.sum(diff**2)

        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


def one_hot(labels, n_classes=None):
    labels = np.asarray(labels, dtype=int)
    if n_classes is None:
        n_classes = int(labels.max()) + 1
    n_samples = labels.shape[0]
    E = np.zeros((n_samples, n_classes), dtype=float)
    E[np.arange(n_samples), labels] = 1.0
    return E


def knn_graph(X, n_neighbors=5, sigma=None):
    """
    使用高斯核构建对称 kNN 图。
    X: (n_nodes, dim)
    返回 S: (n_nodes, n_nodes)
    """
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]

    # 两两欧式距离平方
    diff = X[:, None, :] - X[None, :, :]
    dist2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist2, np.inf)

    # k 最近邻索引
    knn_idx = np.argsort(dist2, axis=1)[:, :n_neighbors]

    # 自适应 sigma：各点到其 kNN 的距离的均值
    if sigma is None:
        knn_distances = dist2[np.arange(n_samples)[:, None], knn_idx]
        sigma = np.sqrt(np.mean(knn_distances))
        if sigma <= 0 or not np.isfinite(sigma):
            sigma = 1.0

    S = np.zeros((n_samples, n_samples), dtype=float)
    for i in range(n_samples):
        for j in knn_idx[i]:
            if not np.isfinite(dist2[i, j]):
                continue
            w = np.exp(-dist2[i, j] / (sigma ** 2))
            # xi ∈ N(xj) 或 xj ∈ N(xi) 时取最大权重，保证对称
            S[i, j] = max(S[i, j], w)
            S[j, i] = max(S[j, i], w)

    return S


def laplacian(S):
    d = S.sum(axis=1)
    return np.diag(d) - S


# ------------------------------------------------------------
# GOD 中各个变量的更新：W, E, B，以及目标函数
# ------------------------------------------------------------

def update_V_W(W, eps=1e-8):
    """
    计算 l2,1-2 非凸正则中的对角重加权矩阵 V_W（式(24)）。
    """
    row_norm2 = np.sum(W**2, axis=1)           # 每行的 ||w_i||_2^2
    denom = np.maximum(row_norm2, eps)
    diag_vals = 1.0 / (2.0 * denom)
    return np.diag(diag_vals)


def update_W(X_d_n, W, B, E, L_W, L_E, beta, alpha, gamma, eps=1e-8):
    """
    按照式(23)(25) 对 W 做一次乘法更新：
        W_ij <- W_ij * (正梯度项 / 负梯度项)
    X_d_n : (d, n)  （论文记号中的 X）
    W     : (d, c)
    B     : (c, c)
    E     : (n, c)
    L_W   : (d, d) 特征图拉普拉斯
    L_E   : (n, n) E 上的拉普拉斯（局部保持项用）
    """
    d, n = X_d_n.shape
    X = X_d_n

    # Frobenius 范数平方 Tr(WW^T)
    tr_WWT = np.sum(W**2)

    # X E B^T  (d x n)(n x c)(c x c) = (d x c)
    XEBt = X @ (E @ B.T)

    V_W = update_V_W(W, eps=eps)

    # XX^T W
    XXt = X @ X.T               # (d x d)
    XXtW = XXt @ W

    # X L_E X^T W
    XLEXtW = (X @ L_E @ X.T) @ W

    # L_W W
    LW_W = L_W @ W

    # 分子：XEB^T + 0.5 * beta * (Tr(WW^T))^{-1/2} * W
    tr_term = np.sqrt(max(tr_WWT, eps))
    num = XEBt + 0.5 * beta * (1.0 / tr_term) * W

    # 分母：XX^T W + beta V_W W + gamma X L_E X^T W + alpha L_W W
    denom = XXtW + beta * (V_W @ W) + gamma * XLEXtW + alpha * LW_W
    denom = np.maximum(denom, eps)

    W_new = W * (num / denom)
    # 保证非负，避免 0 锁死
    W_new = np.maximum(W_new, eps)
    return W_new


def update_E(A, C, E_init, n_iter=5):
    """
    对应 Algorithm 1：在 Stiefel 流形上解
        min_{E^T E = I} Tr(E^T A E) - Tr(E^T C)
    用 A_tilde = mu I - A 的形式转成
        max Tr(E^T P)，P = A_tilde E + C/2
    然后用 Procrustes（SVD）得到 E = U V^T。
    A      : (n, n)，这里取 alpha * L
    C      : (n, c)，这里取 2(X W B + eta Z)
    E_init : (n, c)
    """
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    E = np.asarray(E_init, dtype=float)

    n, c = E.shape

    # 先把初始 E 正交化
    U, _, Vt = np.linalg.svd(E, full_matrices=False)
    E = U @ Vt

    # 估计 mu > 最大特征值，保证 A_tilde 正定
    try:
        eigvals = np.linalg.eigvalsh(A)
        mu = float(eigvals.max()) + 1e-3
    except np.linalg.LinAlgError:
        # 回退：用 trace/n 当粗略上界
        mu = float(np.trace(A) / A.shape[0] + 1.0)

    A_tilde = mu * np.eye(A.shape[0]) - A

    for _ in range(n_iter):
        # P = A_tilde E + C/2
        P = A_tilde @ E + 0.5 * C
        # 解 max_{E^T E = I} Tr(E^T P) => E = U V^T
        U, _, Vt = np.linalg.svd(P, full_matrices=False)
        E = U @ Vt

    return E


def update_B(X_d_n, W, E):
    """
    更新 B 的正交 Procrustes 解：
        min_{B^T B = I} || W^T X - B E^T ||_F^2
    等价于最大化 Tr(B E^T X^T W)，
    解为: B = U V^T，其中 U Σ V^T = E^T X^T W。
    """
    X = X_d_n
    # M = E^T X^T W = (c x n)(n x d)(d x c) = (c x c)
    M = E.T @ X.T @ W
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    B = U @ Vt
    return B


def l21_norm(W, eps=1e-12):
    row_norm = np.sqrt(np.sum(W**2, axis=1) + eps)
    return np.sum(row_norm)


def frob_norm(W):
    return np.sqrt(np.sum(W**2))


def objective(X_d_n, W, B, E, Z, L_W, L, L_E, alpha, beta, gamma, eta):
    """
    目标函数 (18)：
      ||W^T X - B E^T||_F^2
      + beta ( ||W||_{2,1} - ||W||_2 )
      + alpha ( Tr(W^T L_W W) + Tr(E^T L E) )
      + gamma || W^T X L_E X^T W ||_F^2
      + eta ||Z - E||_F^2
    """
    X = X_d_n

    # 重构项
    WT_X = W.T @ X        # (c, n)
    B_Et = B @ E.T        # (c, n)
    recon = np.sum((WT_X - B_Et)**2)

    # 非凸稀疏项
    l21 = l21_norm(W)
    l2 = frob_norm(W)
    sparse_term = beta * (l21 - l2)

    # 图嵌入 / 几何结构项
    term_W_graph = alpha * np.trace(W.T @ L_W @ W)
    term_E_graph = alpha * np.trace(E.T @ L @ E)

    # 局部结构保持项
    M = W.T @ X           # (c, n)
    N = M @ L_E @ X.T @ W # (c, c)
    local_term = gamma * np.sum(N**2)

    # Z-E 近似项
    ze_term = eta * np.sum((Z - E)**2)

    return recon + sparse_term + term_W_graph + term_E_graph + local_term + ze_term


# ------------------------------------------------------------
# GOD 核心：图嵌入聚类标签正交分解的特征选择 (对应 Algorithm 2)
# ------------------------------------------------------------

def god_feature_selection(
    X,
    n_clusters,
    n_selected_features,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    eta=1.0,
    k_neighbors=5,
    max_iter=50,
    tol=1e-4,
    inner_iter_E=5,
    random_state=None,
):
    """
    图嵌入正交分解 (GOD) 的核心特征选择过程。
    X : (n_samples, n_features)
    n_clusters : 簇个数 c
    n_selected_features : 选取特征数 m
    alpha, beta, gamma, eta : 论文中的四个平衡参数
    k_neighbors : 构图时的 k
    max_iter : GOD 迭代次数
    tol : 基于 W 的相对收敛判据
    inner_iter_E : 更新 E 时的内循环次数（Algorithm 1）
    random_state : 随机种子
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    d = n_features
    n = n_samples
    m = min(n_selected_features, n_features)

    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if m <= 0:
        raise ValueError("n_selected_features must be positive")

    rng = np.random.RandomState(random_state)

    # 论文记号中的 X : d x n
    X_d_n = X.T

    # (1) 基于样本的图 S, L  (n x n)
    S_samples = knn_graph(X, n_neighbors=k_neighbors)
    L = laplacian(S_samples)

    # (2) 基于特征的图 S_W, L_W (d x d)，用 X 的每一行（特征向量）构图
    S_W = knn_graph(X_d_n, n_neighbors=k_neighbors)
    L_W = laplacian(S_W)

    # (3) 用 KMeans 在样本空间得到初始 E (n x c)
    labels0 = kmeans(X, n_clusters=n_clusters, random_state=random_state)
    E = one_hot(labels0, n_classes=n_clusters)

    # (4) 在 E 上再建图（局部保持项用）S_E, L_E (n x n)
    S_E = knn_graph(E, n_neighbors=k_neighbors)
    L_E = laplacian(S_E)

    # (5) 初始化 W (d x c), B (c x c), Z (n x c)
    W = np.abs(rng.randn(d, n_clusters)) + 1e-6
    Q, _ = np.linalg.qr(rng.randn(n_clusters, n_clusters))  # 随机正交矩阵
    B = Q
    Z = np.maximum(E, 0.0)

    curve = []

    for it in range(max_iter):
        W_old = W.copy()

        # ---- 更新 W (式(25)) ----
        W = update_W(X_d_n, W, B, E, L_W, L_E, beta=beta, alpha=alpha, gamma=gamma)

        # ---- 更新 E (Algorithm 1，对应式(26)-(28)) ----
        A = alpha * L                           # (n x n)
        # 这里把 eta 也体现在 C 中：C = 2(X W B + eta Z)
        C = 2.0 * (X @ W @ B + eta * Z)         # (n x c)
        E = update_E(A, C, E, n_iter=inner_iter_E)

        # ---- 更新 Z (式(31)) ----
        Z = np.maximum(E, 0.0)

        # ---- 更新 B (式(19)(20)) ----
        B = update_B(X_d_n, W, E)

        # ---- 计算目标函数值 (式(18)) ----
        obj = objective(X_d_n, W, B, E, Z, L_W, L, L_E, alpha, beta, gamma, eta)
        curve.append(obj)

        # ---- 收敛判据：W 的相对变化 ----
        rel_change = np.linalg.norm(W - W_old) / (np.linalg.norm(W_old) + 1e-8)
        # print(f"Iter {it:03d}: obj={obj:.4e}, rel_change={rel_change:.3e}")
        if rel_change < tol:
            break

    # 行 l2 范数作为特征重要性得分
    scores = np.linalg.norm(W, axis=1)
    idx_sorted = np.argsort(-scores)
    selected_idx = idx_sorted[:m]

    Gbin = np.zeros(n_features, dtype=int)
    Gbin[selected_idx] = 1

    return Gbin, np.array(curve), W


# ------------------------------------------------------------
# 对外接口：fs(xtrain, xvalid, ytrain, yvalid, opts)
# ------------------------------------------------------------

def fs(xtrain, xvalid, ytrain, yvalid, opts):
    """
    外层 wrapper，签名按你的要求：

        def fs(xtrain, xvalid, ytrain, yvalid, opts):
            return {'sf': Gbin, 'c': curve, 'nf': num_feat}

    参数
    ----
    xtrain : (n_samples, n_features)，训练数据
    xvalid : 验证集（本算法为无监督，未使用；保留接口兼容）
    ytrain, yvalid : 标签（本算法不使用标签，只在 n_clusters 未指定时
                     用 ytrain 推出簇数）
    opts : dict，主要键：
        - 'nf' 或 'num_features' : 选取特征数 m
        - 'n_clusters' (可选)    : 簇数 c；若缺省且 ytrain 不为 None，则用
                                  len(np.unique(ytrain))
        - 'alpha', 'beta', 'gamma', 'eta' (可选) : GOD 的 4 个权重
        - 'k_neighbors', 'max_iter', 'tol', 'inner_iter_E', 'random_state' (可选)

    返回
    ----
    dict:
        'sf' : 二值选择向量 (n_features,)
        'c'  : 目标函数值曲线 (n_iter,)
        'nf' : 实际选中特征数
    """
    Xtr = np.asarray(xtrain, dtype=float)
    n_samples, n_features = Xtr.shape

    # 选取特征数
    if 'nf' in opts:
        n_selected = int(opts['nf'])
    elif 'num_features' in opts:
        n_selected = int(opts['num_features'])
    else:
        n_selected = n_features   # 默认保留全部

    # 簇数：优先从 opts['n_clusters']，否则用 ytrain 推断
    if 'n_clusters' in opts:
        n_clusters = int(opts['n_clusters'])
    else:
        if ytrain is None:
            raise ValueError("n_clusters not provided in opts and ytrain is None.")
        ytr = np.asarray(ytrain)
        n_clusters = int(len(np.unique(ytr)))

    alpha = float(opts.get('alpha', 1.0))
    beta = float(opts.get('beta', 1.0))
    gamma = float(opts.get('gamma', 1.0))
    eta = float(opts.get('eta', 1.0))
    k_neighbors = int(opts.get('k_neighbors', 5))
    max_iter = int(opts.get('max_iter', 50))
    tol = float(opts.get('tol', 1e-4))
    inner_iter_E = int(opts.get('inner_iter_E', 5))
    random_state = opts.get('random_state', None)

    Gbin, curve, W = god_feature_selection(
        Xtr,
        n_clusters=n_clusters,
        n_selected_features=n_selected,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        eta=eta,
        k_neighbors=k_neighbors,
        max_iter=max_iter,
        tol=tol,
        inner_iter_E=inner_iter_E,
        random_state=random_state,
    )

    num_feat = int(Gbin.sum())

    return {'sf': Gbin, 'c': curve, 'nf': num_feat}
