import math
import numpy as np
from numpy.random import default_rng

from Function import cross_validation_value


def _encode_labels(y):
    _, inv = np.unique(y, return_inverse=True)
    return inv


def _entropy_from_counts(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return -np.sum(p * np.log2(p))


def _discretize_feature(col, num_bins):
    n = len(col)
    order = np.argsort(col)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(n)
    bins = (ranks * num_bins) // n
    bins[bins >= num_bins] = num_bins - 1
    return bins.astype(int)


def _discretize_matrix(X, num_bins):
    n, d = X.shape
    disc = np.empty((n, d), dtype=int)
    for j in range(d):
        disc[:, j] = _discretize_feature(X[:, j], num_bins)
    return disc


def _su_from_discrete(a, b, a_bins, b_bins):
    joint = np.zeros((a_bins, b_bins), dtype=int)
    np.add.at(joint, (a, b), 1)
    hx = _entropy_from_counts(joint.sum(axis=1))
    hy = _entropy_from_counts(joint.sum(axis=0))
    hxy = _entropy_from_counts(joint.ravel())
    ig = hx + hy - hxy
    if hx + hy == 0:
        return 0.0
    return 2.0 * ig / (hx + hy)


def _su_feature_label(disc_X, y_disc, num_bins, num_classes):
    n_features = disc_X.shape[1]
    su = np.zeros(n_features)
    for j in range(n_features):
        su[j] = _su_from_discrete(disc_X[:, j], y_disc, num_bins, num_classes)
    return su


def _evaluate(mask, xtrain, xvalid, ytrain, yvalid, alpha, opts):
    num_feat = int(mask.sum())
    if num_feat == 0:
        return 1.0
    error = cross_validation_value(xtrain, xvalid, ytrain, yvalid, mask, opts)
    return alpha * error + (1.0 - alpha) * (num_feat / mask.size)


def _initialize_population(N, D, div_ratio, sorted_idx, rng):
    first_size = max(1, int(round(N * div_ratio)))
    positions = np.zeros((N, D), dtype=float)
    velocities = rng.uniform(-1.0, 1.0, size=(N, D))

    for i in range(first_size):
        length = int(math.ceil((i + 1) * D / N))
        sel = sorted_idx[:length]
        positions[i, sel] = rng.random(length)

    for i in range(first_size, N):
        positions[i] = rng.random(D)

    return positions, velocities


def _compute_learning_probabilities(fitness):
    N = len(fitness)
    rank_idx = np.argsort(np.argsort(fitness))  # 0 for best
    exp10 = math.exp(10)
    probs = np.zeros(N)
    for i in range(N):
        r = rank_idx[i]
        probs[i] = 0.05 + 0.45 * math.exp(10 * r / max(1, N - 1)) / (exp10 - 1.0)
    return probs


def fs(xtrain, xvalid, ytrain, yvalid, opts=None):
    """
    Leader-adaptive PSO with dimensionality reduction (LAPSO-DR).

    Key options (all optional):
        N: population size (default 50)
        T: max iterations (default 100)
        thres: binarization threshold (default 0.5)
        c1, c2: acceleration coefficients (default 1.5, 1.5)
        w_max, w_min: inertia bounds (default 0.9, 0.4)
        div_ratio: population division ratio (default 0.7 -> first part)
        lapsodr_m: stagnation threshold for leader refresh (default 7)
        num_bins: discretization bins for SU (default 10)
        phi: weight for classification error in fitness (default 0.9 per paper)
    """
    if opts is None:
        opts = {}

    rng = default_rng(opts.get("random_seed"))
    N = int(opts.get("N", 50))
    T = int(opts.get("T", 100))
    thres = float(opts.get("thres", 0.5))
    c1 = float(opts.get("c1", 1.5))
    c2 = float(opts.get("c2", 1.5))
    w_max = float(opts.get("w_max", 0.9))
    w_min = float(opts.get("w_min", 0.4))
    div_ratio = float(opts.get("div_ratio", 0.7))
    stagnation_limit = int(opts.get("lapsodr_m", 7))
    num_bins = int(opts.get("num_bins", 10))
    alpha = float(opts.get("phi", 0.9))

    dim = xtrain.shape[1]
    X_all = np.vstack((xtrain, xvalid))
    y_all = np.concatenate((ytrain, yvalid))
    y_disc = _encode_labels(y_all)
    num_classes = int(y_disc.max() + 1)

    disc_X = _discretize_matrix(X_all, num_bins)
    su_label = _su_feature_label(disc_X, y_disc, num_bins, num_classes)
    sorted_idx = np.argsort(-su_label)

    xtrain_ord = xtrain
    xvalid_ord = xvalid

    top_k = max(1, int(0.3 * dim))
    top_feat_idx = sorted_idx[:top_k]
    su_cache = {}

    def get_pair_su(fa, fb):
        key = (int(min(fa, fb)), int(max(fa, fb)))
        if key not in su_cache:
            su_cache[key] = _su_from_discrete(
                disc_X[:, key[0]], disc_X[:, key[1]], num_bins, num_bins
            )
        return su_cache[key]

    positions, velocities = _initialize_population(N, dim, div_ratio, sorted_idx, rng)
    fitness = np.zeros(N)
    pbest_pos = positions.copy()
    pbest_fit = np.full(N, np.inf)
    stagnation = np.zeros(N, dtype=int)
    leaders = np.zeros(N, dtype=int)

    for i in range(N):
        mask = (positions[i] > thres).astype(int)
        fitness[i] = _evaluate(mask, xtrain_ord, xvalid_ord, ytrain, yvalid, alpha, opts)
        pbest_pos[i] = positions[i]
        pbest_fit[i] = fitness[i]
    gbest_idx = int(np.argmin(pbest_fit))
    gbest_fit = float(pbest_fit[gbest_idx])
    curve = []

    for t in range(T):
        h = max(0.1, min(0.5, -0.4 * (t + 1) / max(1, T) + 0.5))
        elite_size = max(1, int(round(h * N)))
        elite_idx = np.argsort(fitness)[:elite_size]

        # Leader-adaptive strategy
        for i in range(N):
            if t == 0 or stagnation[i] > stagnation_limit:
                leaders[i] = rng.choice(elite_idx)
                stagnation[i] = 0

        # Inter-particle learning (CLPSO-style with rank-based probability)
        learning_probs = _compute_learning_probabilities(fitness)
        exemplars = np.full((N, dim), -1, dtype=int)
        for i in range(N):
            if rng.random() < learning_probs[i]:
                for d in range(dim):
                    r1, r2 = rng.choice(N, size=2, replace=False)
                    exemplar = r1 if pbest_fit[r1] < pbest_fit[r2] else r2
                    exemplars[i, d] = exemplar
            else:
                exemplars[i, :] = i

        w = w_max - (w_max - w_min) * (t / max(1, T - 1))
        for i in range(N):
            leader_pos = pbest_pos[leaders[i]]
            exemplar_idx = exemplars[i]
            exemplar_pos = pbest_pos[exemplar_idx, np.arange(dim)]
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (exemplar_pos - positions[i])
                + c2 * r2 * (leader_pos - positions[i])
            )
            positions[i] = np.clip(positions[i] + velocities[i], 0.0, 1.0)

        # Evaluate population
        for i in range(N):
            mask = (positions[i] > thres).astype(int)
            fit_val = _evaluate(mask, xtrain_ord, xvalid_ord, ytrain, yvalid, alpha, opts)
            fitness[i] = fit_val
            if fit_val < pbest_fit[i]:
                pbest_fit[i] = fit_val
                pbest_pos[i] = positions[i].copy()
                stagnation[i] = 0
            else:
                stagnation[i] += 1

        # Dimensionality reduction on elites (AMB-based)
        for idx in elite_idx:
            pos_vec = positions[idx].copy()
            mask = (pos_vec > thres).astype(int)
            candidate_feats = top_feat_idx
            vals = pos_vec[candidate_feats]
            q1_mask = vals > thres
            q1_feats = candidate_feats[q1_mask]
            if q1_feats.size == 0:
                avg_su = su_label[candidate_feats].mean()
            else:
                avg_su = su_label[q1_feats].mean()
            # delete redundant features
            for a_i, fa in enumerate(q1_feats):
                for fb in q1_feats[a_i + 1 :]:
                    if get_pair_su(fa, fb) >= su_label[fb]:
                        pos_vec[fb] = 0.0
            # add potentially useful unselected features
            for fk in candidate_feats[~q1_mask]:
                if su_label[fk] > avg_su:
                    pos_vec[fk] = 1.0

            new_mask = (pos_vec > thres).astype(int)
            new_fit = _evaluate(new_mask, xtrain_ord, xvalid_ord, ytrain, yvalid, alpha, opts)
            if new_fit < fitness[idx]:
                positions[idx] = pos_vec
                fitness[idx] = new_fit
                if new_fit < pbest_fit[idx]:
                    pbest_fit[idx] = new_fit
                    pbest_pos[idx] = pos_vec.copy()

        gbest_idx = int(np.argmin(pbest_fit))
        gbest_fit = float(pbest_fit[gbest_idx])
        curve.append(gbest_fit)

    best_mask = (pbest_pos[gbest_idx] > thres).astype(int)
    num_feat = int(best_mask.sum())
    return {"sf": best_mask, "c": np.array(curve), "nf": num_feat}
