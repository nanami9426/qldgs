#!/usr/bin/env python3
"""
Python port of the MATLAB pipeline defined in optimization.txt.
The goal is to mirror the original logic and produce equivalent results.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment

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

    G = np.zeros((nClass, nClass))
    for i, lab1 in enumerate(Label):
        for j, lab2 in enumerate(Label):
            G[i, j] = np.sum((L1 == lab1) & (L2 == lab2))
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
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i, j] = np.sum((L1 == Label1[i]) & (L2 == Label2[j]))

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

                E = np.zeros((n, k))
                E[np.arange(n), label - 1] = 1
                center = (E.T @ X) / (E.sum(axis=0)[:, None] + EPS)
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

                E = np.zeros((n, k))
                E[np.arange(n), label - 1] = 1
                center = (E.T @ X) / (E.sum(axis=0)[:, None] + EPS)
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
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nFeat, nSamp = X.shape
    numerator = X @ E @ B.T + 0.5 * beta * (np.trace(W @ W.T) ** -0.5) * W
    denominator = (X @ X.T @ W) + beta * (Da @ W) + gamma * (X @ Ls @ X.T @ W) + alpha * (Lw @ W) + EPS
    W = W * (numerator / denominator)

    T1 = E.T @ X.T @ W
    U_b, _, Vh_b = np.linalg.svd(T1, full_matrices=True)
    B = Vh_b.T @ U_b.T

    AA = alpha * Ls
    try:
        eigvals = np.linalg.eigvalsh(AA)
    except np.linalg.LinAlgError:
        eigvals = np.linalg.eigvals(AA).real
    u1 = float(np.max(eigvals))
    AA1 = u1 * np.eye(nSamp) - AA
    B1 = 2 * (X.T @ W @ B) + 2 * delta * Z
    P = AA1 @ E + 0.5 * B1
    U_m, _, Vh_m = np.linalg.svd(P, full_matrices=True)
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
    m: int,
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
            best_W, X1, best_E, best_B, best_Z, Da, Ls, Lw, n_class, pop[i, 0], pop[i, 1], pop[i, 2], pop[i, 3]
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
            if np.any(V[j, :] > Vmax):
                V[j, :] = Vmax
            if np.any(V[j, :] < Vmin):
                V[j, :] = Vmin
            pop[j, :] = pop[j, :] + V[j, :]
            pop[j, :] = np.minimum(np.maximum(pop[j, :], lb), ub)

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


def main():
    tic = time.time()

    data = loadmat("colon.mat")
    fea = data["fea"]
    gnd = data["gnd"].ravel()

    featurenumset = 200
    NIter = 3
    m = featurenumset
    nClass1 = len(np.unique(gnd))
    SearchAgents_no = 20
    lb = 1e-8
    ub = 1e8
    dim = 4
    Vmax = 0.1 * ub
    Vmin = 0.1 * lb

    fun1 = []
    funx = []
    for _ in range(20):
        X_new, W, idx, Best_pos, Best_score, curve, curve1 = GOD_cPSO_optimization(
            fea.T, nClass1, m, NIter, SearchAgents_no, lb, ub, dim, Vmax, Vmin
        )
        fun1.append(Best_score)
        funx.append(Best_pos)
        print(Best_score)
    fun1 = np.array(fun1)
    funx = np.array(funx)
    b = np.array([np.min(fun1), np.max(fun1), np.mean(fun1), np.std(fun1, ddof=0)])
    c = np.mean(funx, axis=0)

    X_new = X_new.T
    resualt = []
    for _ in range(40):
        label, _, _, _, _ = litekmeans(X_new, nClass1, max_iter=100, replicates=10)
        newres = bestMap(gnd, label)
        AC = np.sum(gnd == newres) / len(gnd)
        MIhat = MutualInfo(gnd, label)
        resualt.append([AC, MIhat])
    resualt = np.array(resualt)

    MEAN = np.zeros((2, 2))
    STD = np.zeros((2, 2))
    BEST = np.zeros((2, 1))
    for j in range(2):
        a = resualt[:, j]
        temp = []
        for i in range(len(a)):
            if i < len(a) - 18:
                temp.append(np.sum(a[i : i + 20]))
        temp = np.array(temp)
        f_idx = int(np.argmax(temp))
        e = temp[f_idx] / 20.0
        f_matlab = f_idx + 1  # record MATLAB-style index
        MEAN[j, :] = [e, f_matlab]
        STD[j, :] = np.std(resualt[f_idx : f_idx + 20, j])
        rr = np.sort(resualt[:, j])
        BEST[j, 0] = rr[-1]

    print("算法运行完毕！")
    print("以下是AP_OCLGR算法运行得到的AR10P数据集的ACC与NMI值：")
    print("ACC±STD%%:")
    print(f"{MEAN[0,0]*100:.2f}\t", end="")
    print(f"{STD[0,0]*100:.2f}")
    print("\nNMI±STD%%:")
    print(f"{MEAN[1,0]*100:.2f}\t", end="")
    print(f"{STD[1,0]*100:.2f}")
    print()
    print(f"Elapsed: {time.time() - tic:.2f}s")


if __name__ == "__main__":
    main()
