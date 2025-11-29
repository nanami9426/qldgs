#!/usr/bin/env python3
"""
Python port of the MATLAB pipeline defined in optimization.txt.
The goal is to mirror the original logic and produce equivalent results.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from Function import Fun

EPS = np.finfo(float).eps


def EuDist2(fea_a: np.ndarray, fea_b: Optional[np.ndarray] = None, bSqrt: bool = True) -> np.ndarray:
    """
    Compute the (squared) Euclidean distance matrix.
    Mirrors the MATLAB EuDist2 implementation used in optimization.txt.
    """
    if fea_b is None:
        aa = np.sum(fea_a * fea_a, axis=1)
        ab = fea_a @ fea_a.T
        D = aa[:, None] + aa[None, :] - 2 * ab
        D[D < 0] = 0
        if bSqrt:
            D = np.sqrt(D)
        return np.maximum(D, D.T)

    aa = np.sum(fea_a * fea_a, axis=1)
    bb = np.sum(fea_b * fea_b, axis=1)
    ab = fea_a @ fea_b.T
    D = aa[:, None] + bb[None, :] - 2 * ab
    D[D < 0] = 0
    if bSqrt:
        D = np.sqrt(D)
    return D


def NormalizeFea(fea: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(fea, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return fea / norms


def constructW(fea: np.ndarray, options: Dict) -> np.ndarray:
    """
    Graph construction function translated from constructW.m.
    Only the logic exercised by optimization.txt is required, but the full
    control flow is preserved for fidelity.
    """
    if options is None:
        options = {}

    neighbor_mode = options.get("NeighborMode", "KNN").lower()
    weight_mode = options.get("WeightMode", "HeatKernel").lower()
    k = int(options.get("k", 5))
    t = options.get("t", None)
    b_normalized = bool(options.get("bNormalized", False))
    b_self_connected = bool(options.get("bSelfConnected", False))
    b_true_knn = bool(options.get("bTrueKNN", False))

    b_binary = weight_mode == "binary"
    b_cosine = weight_mode == "cosine"

    if t is None and weight_mode == "heatkernel":
        nSmp = fea.shape[0]
        if nSmp > 3000:
            idx = np.random.choice(nSmp, 3000, replace=False)
            D = EuDist2(fea[idx, :])
        else:
            D = EuDist2(fea)
        t = np.mean(D)

    if neighbor_mode == "supervised":
        raise ValueError("Supervised graph mode is not needed in this pipeline.")

    if b_cosine and not b_normalized:
        fea = NormalizeFea(fea)

    nSmp = fea.shape[0]

    if neighbor_mode == "knn" and k > 0:
        if b_cosine:
            dist = fea @ fea.T
            # For cosine similarity, larger values indicate closer neighbors.
            idx = np.argpartition(-dist, range(k + 1), axis=1)[:, : k + 1]
            weights = dist[np.arange(nSmp)[:, None], idx]
            if not b_binary:
                dump = weights
            else:
                dump = np.ones_like(weights)
        else:
            dist = EuDist2(fea, None, bSqrt=False)
            idx = np.argpartition(dist, range(k + 1), axis=1)[:, : k + 1]
            weights = dist[np.arange(nSmp)[:, None], idx]
            if not b_binary:
                dump = np.exp(-weights / (2 * (t ** 2)))
            else:
                dump = np.ones_like(weights)

        G = np.zeros((nSmp, nSmp))
        row_ids = np.repeat(np.arange(nSmp), k + 1)
        col_ids = idx.flatten()
        G[row_ids, col_ids] = dump.flatten()
        if b_binary:
            G[G != 0] = 1
        if not b_self_connected:
            np.fill_diagonal(G, 0)
        if not b_true_knn:
            G = np.maximum(G, G.T)
        return G

    # Complete graph branch (k == 0) or other modes fall back here.
    if neighbor_mode == "knn" and k == 0:
        if weight_mode == "binary":
            raise ValueError("Binary weight cannot be used for complete graph.")
        if weight_mode == "heatkernel":
            W = EuDist2(fea, None, bSqrt=False)
            W = np.exp(-W / (2 * (t ** 2)))
        elif weight_mode == "cosine":
            normfea = NormalizeFea(fea)
            W = normfea @ normfea.T
        else:
            raise ValueError("Unknown WeightMode.")
        if not b_self_connected:
            np.fill_diagonal(W, 0)
        return np.maximum(W, W.T)

    raise ValueError("Unsupported NeighborMode.")


def MutualInfo(L1: np.ndarray, L2: np.ndarray) -> float:
    L1 = L1.ravel()
    L2 = L2.ravel()
    if L1.shape != L2.shape:
        raise ValueError("size(L1) must == size(L2)")

    Label = np.unique(L1)
    nClass = len(Label)

    Label2 = np.unique(L2)
    nClass2 = len(Label2)

    # Smooth if class counts mismatch.
    if nClass2 < nClass:
        L1 = np.concatenate([L1, Label])
        L2 = np.concatenate([L2, Label])
    elif nClass2 > nClass:
        L1 = np.concatenate([L1, Label2])
        L2 = np.concatenate([L2, Label2])

    Label = np.unique(L1)
    nClass = len(Label)

    idx1 = np.searchsorted(Label, L1)
    idx2 = np.searchsorted(Label, L2)
    G = np.zeros((nClass, nClass))
    np.add.at(G, (idx1, idx2), 1)
    sumG = np.sum(G)

    P1 = np.sum(G, axis=1) / sumG
    P2 = np.sum(G, axis=0) / sumG

    if np.any(P1 == 0) or np.any(P2 == 0):
        raise ValueError("Smooth fail!")

    H1 = np.sum(-P1 * np.log2(P1))
    H2 = np.sum(-P2 * np.log2(P2))
    P12 = G / sumG
    PPP = P12 / (P2.reshape(1, -1) * P1.reshape(-1, 1))
    PPP[np.abs(PPP) < 1e-12] = 1
    MI = np.sum(P12 * np.log2(PPP))
    MIhat = MI / max(H1, H2)
    return float(np.real(MIhat))


def hungarian(A: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Python replacement for hungarian.m using scipy.optimize.linear_sum_assignment.
    Returns (C, T) where C is the assignment vector (column -> row).
    """
    row_ind, col_ind = linear_sum_assignment(A)
    C = np.zeros(A.shape[1], dtype=int)
    for r, c in zip(row_ind, col_ind):
        C[c] = r + 1  # MATLAB is 1-based; match bestMap expectations.
    T = float(A[row_ind, col_ind].sum())
    return C, T


def bestMap(L1: np.ndarray, L2: np.ndarray) -> np.ndarray:
    L1 = L1.ravel()
    L2 = L2.ravel()
    if L1.shape != L2.shape:
        raise ValueError("size(L1) must == size(L2)")

    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)

    nClass = max(nClass1, nClass2)
    idx1 = np.searchsorted(Label1, L1)
    idx2 = np.searchsorted(Label2, L2)
    G = np.zeros((nClass, nClass))
    np.add.at(G, (idx1, idx2), 1)

    c, _ = hungarian(-G)
    newL2 = np.zeros_like(L2)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i] - 1]
    return newL2


def accuracy(YPred: np.ndarray, target: np.ndarray, classes: Iterable) -> Tuple[float, np.ndarray]:
    classes = np.array(list(classes))
    pred_idx = np.argmax(YPred, axis=1)
    prediction = classes[pred_idx]
    score = float(np.sum(prediction == target) / target.size)
    return score, prediction


def litekmeans(
    X: np.ndarray,
    k: int,
    distance: str = "sqeuclidean",
    start: str | np.ndarray = "sample",
    max_iter: int = 100,
    replicates: int = 1,
    clustermaxiter: int = 10,
) -> Tuple[np.ndarray, np.ndarray, bool, np.ndarray, np.ndarray]:
    """
    Numpy implementation of litekmeans.m.
    Returns (label, center, bCon, sumD, D).
    """
    if k <= 0 or X.shape[0] < k:
        raise ValueError("k must be a positive integer less than the number of samples.")

    n, p = X.shape
    if isinstance(start, np.ndarray):
        if start.ndim == 2 and start.shape[1] == p:
            center_init = start
        elif start.ndim == 1 and start.size == k:
            center_init = X[start, :]
        else:
            raise ValueError("Invalid start matrix.")
        start_mode = "numeric"
    else:
        start_mode = start.lower()
        center_init = None

    if center_init is not None:
        replicates = 1

    bestlabel = None
    bestcenter = None
    bestsumD = None
    bestD = None
    bCon = False

    for rep in range(replicates):
        if start_mode == "sample":
            center = X[np.random.choice(n, k, replace=False), :]
        elif start_mode == "cluster":
            subset = X[np.random.choice(n, max(1, int(math.floor(0.1 * n))), replace=False), :]
            label_sub, center, _, _, _ = litekmeans(
                subset, k, distance=distance, start="sample", max_iter=clustermaxiter, replicates=1
            )
        elif start_mode == "numeric":
            center = center_init.copy()
        else:
            raise ValueError("Unknown start mode.")

        label = np.ones(n, dtype=int)
        last = np.zeros_like(label)
        it = 0

        if distance.lower() == "sqeuclidean":
            while np.any(label != last) and it < max_iter:
                last = label.copy()
                bb = np.sum(center * center, axis=1)
                ab = X @ center.T
                D = bb.reshape(1, -1) - 2 * ab
                label = np.argmin(D, axis=1) + 1  # 1-based labels

                ll = np.unique(label)
                if len(ll) < k:
                    miss_cluster = [c for c in range(1, k + 1) if c not in ll]
                    aa = np.sum(X * X, axis=1)
                    val = aa + D[np.arange(n), label - 1]
                    idx = np.argsort(val)[::-1]
                    label[idx[: len(miss_cluster)]] = miss_cluster

                counts = np.bincount(label - 1, minlength=k).astype(float)
                center = np.zeros((k, p))
                for cls in range(k):
                    if counts[cls] > 0:
                        center[cls] = X[label == (cls + 1)].sum(axis=0) / counts[cls]
                it += 1
            bCon = it < max_iter
            aa = np.sum(X * X, axis=1)
            bb = np.sum(center * center, axis=1)
            ab = X @ center.T
            D = aa[:, None] + bb.reshape(1, -1) - 2 * ab
            D[D < 0] = 0
            D = np.sqrt(D)
        else:  # cosine
            center = NormalizeFea(center)
            while np.any(label != last) and it < max_iter:
                last = label.copy()
                W = X @ center.T
                label = np.argmax(W, axis=1) + 1
                ll = np.unique(label)
                if len(ll) < k:
                    miss_cluster = [c for c in range(1, k + 1) if c not in ll]
                    val = W[np.arange(n), label - 1]
                    idx = np.argsort(val)
                    label[idx[: len(miss_cluster)]] = miss_cluster

                counts = np.bincount(label - 1, minlength=k).astype(float)
                center = np.zeros((k, p))
                for cls in range(k):
                    if counts[cls] > 0:
                        center[cls] = X[label == (cls + 1)].sum(axis=0) / counts[cls]
                center = NormalizeFea(center)
                it += 1
            bCon = it < max_iter
            W = X @ center.T
            D = 1 - W

        sumD = np.array([np.sum(D[label == (j + 1), j]) for j in range(k)])

        if bestlabel is None or np.sum(sumD) < np.sum(bestsumD):
            bestlabel = label.copy()
            bestcenter = center.copy()
            bestsumD = sumD.copy()
            bestD = D.copy()

    return bestlabel, bestcenter, bCon, bestsumD, bestD


def optimization1(
    W: np.ndarray,
    X: np.ndarray,
    E: np.ndarray,
    B: np.ndarray,
    Z: np.ndarray,
    Da: np.ndarray,
    Ls: np.ndarray,
    Lw: np.ndarray,
    n_class: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    *,
    XTX: Optional[np.ndarray] = None,
    XLsXT: Optional[np.ndarray] = None,
    lambda_max_Ls: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nFeat, nSamp = X.shape
    if XTX is None:
        XTX = X @ X.T
    if XLsXT is None:
        XLsXT = X @ Ls @ X.T
    numerator = X @ E @ B.T + 0.5 * beta * (np.trace(W @ W.T) ** -0.5) * W
    denominator = (XTX @ W) + beta * (Da @ W) + gamma * (XLsXT @ W) + alpha * (Lw @ W) + EPS
    W = W * (numerator / denominator)

    T1 = E.T @ X.T @ W
    U_b, _, Vh_b = np.linalg.svd(T1, full_matrices=False)
    B = Vh_b.T @ U_b.T

    if lambda_max_Ls is None:
        try:
            lambda_max_Ls = float(np.linalg.eigvalsh(Ls).max())
        except np.linalg.LinAlgError:
            lambda_max_Ls = float(np.linalg.eigvals(Ls).real.max())
    AA1 = alpha * (lambda_max_Ls * np.eye(nSamp) - Ls)
    B1 = 2 * (X.T @ W @ B) + 2 * delta * Z
    P = AA1 @ E + 0.5 * B1
    U_m, _, Vh_m = np.linalg.svd(P, full_matrices=False)
    E = U_m[:, :n_class] @ Vh_m[:n_class, :]

    Z = np.maximum(E, 0)

    obj = np.linalg.norm((W.T @ X) - (B @ E.T), ord="fro") ** 2
    obj += alpha * np.trace(W.T @ Lw @ W)
    obj += beta * (np.trace(W.T @ Da @ W) - np.linalg.norm(W, 2))
    obj += gamma * np.trace(W.T @ X @ Ls @ X.T @ W)
    obj += delta * (np.linalg.norm(E - Z, ord="fro") ** 2)
    return abs(float(obj)), W, E, B, Z


def GOD_cPSO_optimization(
    X1: np.ndarray,
    n_class: int,
    m: Optional[int],
    NIter: int,
    sizepop: int,
    lb: float,
    ub: float,
    Dim: int,
    Vmax: float,
    Vmin: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    wmax = 0.9
    wmin = 0.1
    c1 = 2
    c2 = 2

    nFeat, nSamp = X1.shape
    if m is None:
        m = nFeat
    m = int(max(1, min(nFeat, m)))

    best_W = np.ones((nFeat, n_class))
    best_B = np.random.rand(n_class, n_class)
    best_E = np.random.rand(nSamp, n_class)
    best_Z = np.random.rand(nSamp, n_class)

    options = {
        "NeighborMode": "KNN",
        "k": 5,
        "t": 1e4,
        "WeightMode": "Heatkernel",
    }

    S1 = constructW(X1.T, options)
    Ds = np.diag(np.sum(S1, axis=1))
    Ls = Ds - S1

    S2 = constructW(X1, options)
    D2 = np.diag(np.sum(S2, axis=1))
    Lw = D2 - S2

    # Precompute invariant matrices to avoid repeated large multiplications.
    XTX = X1 @ X1.T
    XLsXT = X1 @ Ls @ X1.T
    try:
        lambda_max_Ls = float(np.linalg.eigvalsh(Ls).max())
    except np.linalg.LinAlgError:
        lambda_max_Ls = float(np.linalg.eigvals(Ls).real.max())

    Wi = np.sqrt(np.sum(best_W * best_W, axis=1) + EPS)
    d = 0.5 / Wi
    Da = np.diag(d)

    if np.size(ub) == 1:
        ub = np.full(Dim, ub)
        lb = np.full(Dim, lb)

    Range = np.ones((sizepop, 1)) * (ub - lb)
    pop = np.random.rand(sizepop, Dim) * Range + np.ones((sizepop, 1)) * lb
    V = np.random.rand(sizepop, Dim) * (Vmax - Vmin) + Vmin

    fitness = np.zeros(sizepop)
    Wa = np.zeros((nFeat, n_class, sizepop))
    Ea = np.zeros((nSamp, n_class, sizepop))
    Ba = np.zeros((n_class, n_class, sizepop))
    Za = np.zeros((nSamp, n_class, sizepop))

    for i in range(sizepop):
        fitness[i], Wa[:, :, i], Ea[:, :, i], Ba[:, :, i], Za[:, :, i] = optimization1(
            best_W,
            X1,
            best_E,
            best_B,
            best_Z,
            Da,
            Ls,
            Lw,
            n_class,
            pop[i, 0],
            pop[i, 1],
            pop[i, 2],
            pop[i, 3],
            XTX=XTX,
            XLsXT=XLsXT,
            lambda_max_Ls=lambda_max_Ls,
        )

    bestindex = int(np.argmin(fitness))
    zbest = pop[bestindex, :].copy()
    gbest = pop.copy()
    fitnessgbest = fitness.copy()
    fitnesszbest = float(fitness[bestindex])

    best_W = Wa[:, :, bestindex].copy()
    best_E = Ea[:, :, bestindex].copy()
    best_B = Ba[:, :, bestindex].copy()
    best_Z = Za[:, :, bestindex].copy()

    curve = np.zeros(NIter)
    curve1 = np.zeros((NIter, Dim))

    for iter_idx in range(NIter):
        Wi = np.sqrt(np.sum(best_W * best_W, axis=1) + EPS)
        d = 0.5 / Wi
        Da = np.diag(d)
        w = wmax - (wmax - wmin) * (iter_idx + 1) / NIter
        for j in range(sizepop):
            V[j, :] = w * V[j, :] + c1 * np.random.rand() * (gbest[j, :] - pop[j, :]) + c2 * np.random.rand() * (
                zbest - pop[j, :]
            )
            V[j, :] = np.clip(V[j, :], Vmin, Vmax)
            pop[j, :] = np.clip(pop[j, :] + V[j, :], lb, ub)

            fitness[j], Wa[:, :, j], Ea[:, :, j], Ba[:, :, j], Za[:, :, j] = optimization1(
                best_W,
                X1,
                best_E,
                best_B,
                best_Z,
                Da,
                Ls,
                Lw,
                n_class,
                pop[j, 0],
                pop[j, 1],
                pop[j, 2],
                pop[j, 3],
                XTX=XTX,
                XLsXT=XLsXT,
                lambda_max_Ls=lambda_max_Ls,
            )

            if fitness[j] < fitnessgbest[j]:
                gbest[j, :] = pop[j, :]
                fitnessgbest[j] = fitness[j]
            if fitness[j] < fitnesszbest:
                zbest = pop[j, :].copy()
                fitnesszbest = float(fitness[j])
                best_W = Wa[:, :, j].copy()
                best_E = Ea[:, :, j].copy()
                best_B = Ba[:, :, j].copy()
                best_Z = Za[:, :, j].copy()

        curve[iter_idx] = fitnesszbest
        curve1[iter_idx, :] = zbest

    Best_pos = zbest
    Best_score = fitnesszbest

    score = np.sqrt(np.sum(best_W * best_W, axis=1))
    idx = np.argsort(score)[::-1]
    X_new1 = X1[idx[:m], :]
    return X_new1, best_W, idx, Best_pos, Best_score, curve, curve1


def _candidate_feature_sizes(dim: int, opts: Dict) -> list[int]:
    """
    Build a small set of candidate subset sizes so we can search for the best
    feature count instead of relying on the old fixed featurenumset parameter.
    """
    explicit = opts.get("featurenumset")
    if explicit is None:
        explicit = opts.get("feat_num", opts.get("k", None))
    if explicit is not None:
        try:
            size = int(explicit)
        except (TypeError, ValueError):
            size = dim
        return [max(1, min(dim, size))]

    ratios = opts.get("feature_ratio_list", (0.1, 0.2, 0.3, 0.5))
    sizes = {
        max(1, min(dim, int(math.ceil(dim * float(r))))) for r in ratios if r is not None and r > 0
    }
    sizes.add(max(1, int(math.sqrt(dim))))
    sizes.add(max(1, min(dim, int(dim * 0.25))))
    sizes.add(max(1, min(dim, int(dim * 0.4))))
    sizes = sorted(sizes)
    if not sizes:
        sizes = [max(1, min(dim, int(math.sqrt(dim))))]
    return sizes


def fs(xtrain, xvalid, ytrain, yvalid, opts=None):
    """
    GOD-cPSO feature selector aligned with the project interface.
    Returns a dict with binary selection vector, convergence curve, and count.
    """
    if opts is None:
        opts = {}
    opts = dict(opts)

    seed = opts.get("random_seed", None)
    if seed is not None:
        np.random.seed(seed)

    X_all = np.vstack((xtrain, xvalid))
    y_all = np.concatenate((ytrain, yvalid))
    n_class = len(np.unique(y_all))
    dim = X_all.shape[1]

    NIter = int(opts.get("T", 20))
    sizepop = int(opts.get("N", 20))
    lb = float(opts.get("lb", 1e-8))
    ub = float(opts.get("ub", 1e8))
    dim_param = 4
    Vmax = float(opts.get("Vmax", 0.1 * ub))
    Vmin = float(opts.get("Vmin", 0.1 * lb))

    candidate_sizes = _candidate_feature_sizes(dim, opts)

    _, _, idx, _, _, curve, _ = GOD_cPSO_optimization(
        X_all.T, n_class, None, NIter, sizepop, lb, ub, dim_param, Vmax, Vmin
    )

    best_mask = None
    best_k = None
    best_cost = float("inf")
    opts_eval = dict(opts)
    opts_eval["dim"] = dim
    for k in candidate_sizes:
        mask = np.zeros(dim, dtype=int)
        mask[idx[:k]] = 1
        try:
            cost = Fun(xtrain, xvalid, ytrain, yvalid, mask, opts_eval)
        except Exception:
            cost = float("inf")
        if cost < best_cost:
            best_cost = cost
            best_mask = mask
            best_k = k

    if best_mask is None:
        best_k = candidate_sizes[0]
        best_mask = np.zeros(dim, dtype=int)
        best_mask[idx[:best_k]] = 1

    return {"sf": best_mask, "c": curve, "nf": int(best_k)}
