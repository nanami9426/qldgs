import math
from dataclasses import dataclass

import numpy as np
from numpy.random import default_rng
from sklearn.tree import DecisionTreeClassifier

from Function import Fun


_ZERO_TOL = 1e-12


@dataclass
class SubsetStats:
    error: float
    train_acc: float
    valid_acc: float
    gain: float


class DecisionTreeSubsetEvaluator:
    def __init__(
        self,
        xtrain,
        xvalid,
        ytrain,
        yvalid,
        dim,
        gain_weights,
        random_state=None,
    ):
        self.xtrain = np.nan_to_num(np.asarray(xtrain))
        self.xvalid = np.nan_to_num(np.asarray(xvalid))
        self.ytrain = np.asarray(ytrain).reshape(-1)
        self.yvalid = np.asarray(yvalid).reshape(-1)
        self.dim = dim
        self.gain_weights = gain_weights
        self.random_state = random_state
        self.cache = {}

    def evaluate(self, selected):
        key = tuple(sorted(int(i) for i in selected if 0 <= int(i) < self.dim))
        if key in self.cache:
            return self.cache[key]
        if not key:
            stats = SubsetStats(error=1.0, train_acc=0.0, valid_acc=0.0, gain=0.0)
            self.cache[key] = stats
            return stats

        features = np.fromiter(key, dtype=int)
        clf = DecisionTreeClassifier(random_state=self.random_state)
        clf.fit(self.xtrain[:, features], self.ytrain)
        train_acc = float(clf.score(self.xtrain[:, features], self.ytrain))
        valid_acc = float(clf.score(self.xvalid[:, features], self.yvalid))
        compression = 1.0 - (len(key) / max(1, self.dim))
        generalization = 1.0 - min(1.0, abs(train_acc - valid_acc))
        w1, w2, w3 = self.gain_weights
        gain = w1 * valid_acc + w2 * compression + w3 * generalization
        stats = SubsetStats(
            error=1.0 - valid_acc,
            train_acc=train_acc,
            valid_acc=valid_acc,
            gain=float(gain),
        )
        self.cache[key] = stats
        return stats


def _clean_sparse(q):
    return {k: float(v) for k, v in q.items() if abs(v) > _ZERO_TOL}


def _available_actions(dim, selected, end_state):
    if not selected:
        return np.arange(dim + 1, dtype=int)
    unavailable = np.zeros(dim + 1, dtype=bool)
    unavailable[np.asarray(selected, dtype=int)] = True
    actions = np.flatnonzero(~unavailable)
    if actions[-1] != end_state:
        actions = np.append(actions, end_state)
    return actions


def _q_values(q, state, actions):
    return np.array([q.get((state, int(action)), 0.0) for action in actions], dtype=float)


def _max_available_q(q, state, actions):
    if actions.size == 0:
        return 0.0
    values = _q_values(q, state, actions)
    return float(values.max(initial=0.0))


def _sample_next_action(q, state, selected, dim, end_state, epsilon, rng):
    actions = _available_actions(dim, selected, end_state)
    if rng.random() < epsilon:
        return int(rng.choice(actions))

    values = _q_values(q, state, actions)
    best_value = values.max(initial=0.0)
    best_actions = actions[np.isclose(values, best_value)]
    return int(rng.choice(best_actions))


def _greedy_subset(q, dim, end_state, max_steps, rng):
    selected = []
    selected_set = set()
    state = end_state
    for _ in range(max_steps):
        actions = _available_actions(dim, selected, end_state)
        values = _q_values(q, state, actions)
        best_value = values.max(initial=0.0)
        best_actions = actions[np.isclose(values, best_value)]
        action = int(rng.choice(best_actions))
        if action == end_state:
            break
        if action in selected_set:
            break
        selected.append(action)
        selected_set.add(action)
        state = action
    return tuple(selected)


def _qtable_score(q, evaluator, dim, end_state, max_steps, rng):
    selected = _greedy_subset(q, dim, end_state, max_steps, rng)
    stats = evaluator.evaluate(selected)
    return stats.gain, stats.valid_acc, stats, selected


def _copy_qtables(qtables):
    return [dict(q) for q in qtables]


def _scale_inject(q, q_best, alpha):
    keys = set(q) | set(q_best)
    mixed = {}
    for key in keys:
        value = (1.0 - alpha) * q_best.get(key, 0.0) + alpha * q.get(key, 0.0)
        if abs(value) > _ZERO_TOL:
            mixed[key] = float(value)
    return mixed


def _run_sparse_pso(
    qtables,
    evaluator,
    dim,
    end_state,
    max_steps,
    pso_iterations,
    w,
    c1,
    c2,
    vmin,
    vmax,
    q_clip,
    rng,
):
    particle_count = len(qtables)
    velocities = [dict() for _ in range(particle_count)]
    pbest = _copy_qtables(qtables)
    pbest_scores = np.full(particle_count, -np.inf)
    gbest = dict(qtables[0])
    gbest_score = -np.inf
    gbest_valid = -np.inf

    for i, q in enumerate(qtables):
        score, valid_acc, _, _ = _qtable_score(q, evaluator, dim, end_state, max_steps, rng)
        pbest_scores[i] = score
        if (score > gbest_score) or (np.isclose(score, gbest_score) and valid_acc > gbest_valid):
            gbest = dict(q)
            gbest_score = score
            gbest_valid = valid_acc

    for _ in range(max(1, pso_iterations)):
        for i in range(particle_count):
            keys = set(qtables[i]) | set(velocities[i]) | set(pbest[i]) | set(gbest)
            next_q = {}
            next_v = {}
            for key in keys:
                x = qtables[i].get(key, 0.0)
                vel = velocities[i].get(key, 0.0)
                r1 = rng.random()
                r2 = rng.random()
                vel = (
                    w * vel
                    + c1 * r1 * (pbest[i].get(key, 0.0) - x)
                    + c2 * r2 * (gbest.get(key, 0.0) - x)
                )
                vel = float(np.clip(vel, vmin, vmax))
                value = float(np.clip(x + vel, -q_clip, q_clip))
                if abs(value) > _ZERO_TOL:
                    next_q[key] = value
                if abs(vel) > _ZERO_TOL:
                    next_v[key] = vel
            qtables[i] = next_q
            velocities[i] = next_v

            score, valid_acc, _, _ = _qtable_score(
                qtables[i], evaluator, dim, end_state, max_steps, rng
            )
            if score > pbest_scores[i]:
                pbest_scores[i] = score
                pbest[i] = dict(qtables[i])
            if (score > gbest_score) or (
                np.isclose(score, gbest_score) and valid_acc > gbest_valid
            ):
                gbest = dict(qtables[i])
                gbest_score = score
                gbest_valid = valid_acc

    return _clean_sparse(gbest)


def _episode_indices(ytrain, ratio, rng):
    n_samples = len(ytrain)
    if n_samples <= 1:
        return np.arange(n_samples)
    count = int(round(n_samples * ratio))
    count = max(1, min(n_samples, count))
    return rng.choice(n_samples, size=count, replace=False)


def _fallback_feature(q_best, xtrain, dim, end_state):
    row_values = np.array([q_best.get((end_state, j), 0.0) for j in range(dim)])
    if np.any(np.abs(row_values) > _ZERO_TOL):
        return int(np.argmax(row_values))
    variances = np.var(xtrain, axis=0)
    return int(np.argmax(variances))


def _mask_from_selected(selected, dim, q_best, xtrain, end_state):
    mask = np.zeros(dim, dtype=int)
    valid_selected = [int(i) for i in selected if 0 <= int(i) < dim]
    if valid_selected:
        mask[np.asarray(valid_selected, dtype=int)] = 1
    else:
        mask[_fallback_feature(q_best, xtrain, dim, end_state)] = 1
    return mask


def _project_curve_value(mask, xtrain, xvalid, ytrain, yvalid, opts):
    try:
        return float(Fun(xtrain, xvalid, ytrain, yvalid, mask, opts))
    except Exception:
        return 1.0


def _resize_curve(values, target_len):
    if target_len <= 0:
        target_len = len(values)
    if not values:
        return np.ones((1, target_len), dtype=float)
    arr = np.asarray(values, dtype=float)
    if len(arr) == target_len:
        return arr.reshape(1, -1)
    if len(arr) < target_len:
        padded = np.empty(target_len, dtype=float)
        padded[: len(arr)] = arr
        padded[len(arr) :] = arr[-1]
        return padded.reshape(1, -1)
    indices = np.round(np.linspace(0, len(arr) - 1, target_len)).astype(int)
    return arr[indices].reshape(1, -1)


def fs(xtrain, xvalid, ytrain, yvalid, opts=None):
    if opts is None:
        opts = {}

    xtrain = np.nan_to_num(np.asarray(xtrain))
    xvalid = np.nan_to_num(np.asarray(xvalid))
    ytrain = np.asarray(ytrain).reshape(-1)
    yvalid = np.asarray(yvalid).reshape(-1)

    dim = int(xtrain.shape[1])
    if dim == 0:
        return {"sf": np.zeros(0, dtype=int), "c": np.ones((1, int(opts.get("T", 1)))), "nf": 0}

    seed = opts.get("random_seed")
    rng = default_rng(seed)
    agents = max(1, int(opts.get("eqlfs_agents", opts.get("N", 20))))
    episodes = max(1, int(opts.get("eqlfs_episodes", opts.get("T", 100))))
    episode_ratio = float(opts.get("eqlfs_episode_ratio", 0.7))
    episode_ratio = min(1.0, max(0.05, episode_ratio))
    lr = float(opts.get("eqlfs_lr", 0.1))
    gamma = float(opts.get("eqlfs_gamma", 0.005))
    epsilon = float(opts.get("eqlfs_epsilon", 0.1))
    alpha = float(opts.get("eqlfs_alpha", 0.5))
    pso_iterations = max(1, int(opts.get("eqlfs_pso_iterations", 1)))
    pso_w = float(opts.get("eqlfs_pso_w", 0.9))
    pso_c1 = float(opts.get("eqlfs_pso_c1", 2.0))
    pso_c2 = float(opts.get("eqlfs_pso_c2", 2.0))
    vmin = float(opts.get("eqlfs_vmin", -0.5))
    vmax = float(opts.get("eqlfs_vmax", 0.5))
    q_clip = float(opts.get("eqlfs_q_clip", 1.0))
    default_max_steps = min(dim, 50)
    max_steps = max(1, int(opts.get("eqlfs_max_steps", default_max_steps)))
    max_steps = min(dim, max_steps)
    gain_weights = opts.get("eqlfs_gain_weights", (0.5, 0.3, 0.2))
    if isinstance(gain_weights, str):
        gain_weights = tuple(float(x) for x in gain_weights.split(","))
    if len(gain_weights) != 3:
        gain_weights = (0.5, 0.3, 0.2)
    total_weight = sum(float(w) for w in gain_weights)
    if total_weight <= 0:
        gain_weights = (0.5, 0.3, 0.2)
    else:
        gain_weights = tuple(float(w) / total_weight for w in gain_weights)

    end_state = dim
    qtables = [dict() for _ in range(agents)]
    full_evaluator = DecisionTreeSubsetEvaluator(
        xtrain, xvalid, ytrain, yvalid, dim, gain_weights, random_state=seed
    )

    best_mask = None
    best_valid = -np.inf
    best_gain = -np.inf
    best_q = {}
    best_project_fit = math.inf
    curve_values = []

    for _ in range(episodes):
        episode_idx = _episode_indices(ytrain, episode_ratio, rng)
        episode_evaluator = DecisionTreeSubsetEvaluator(
            xtrain[episode_idx],
            xvalid,
            ytrain[episode_idx],
            yvalid,
            dim,
            gain_weights,
            random_state=seed,
        )

        for q in qtables:
            selected = []
            selected_set = set()
            state = end_state
            last_error = 0.0
            for _ in range(max_steps):
                action = _sample_next_action(
                    q, state, selected, dim, end_state, epsilon, rng
                )
                if action == end_state:
                    break
                if action in selected_set:
                    break

                selected.append(action)
                selected_set.add(action)
                stats = episode_evaluator.evaluate(selected)
                reward = last_error - stats.error
                next_state = action
                next_actions = _available_actions(dim, selected, end_state)
                td_target = reward + gamma * _max_available_q(
                    q, next_state, next_actions
                )
                key = (state, action)
                old_value = q.get(key, 0.0)
                new_value = old_value + lr * (td_target - old_value)
                if abs(new_value) > _ZERO_TOL:
                    q[key] = float(np.clip(new_value, -q_clip, q_clip))
                elif key in q:
                    del q[key]
                last_error = stats.error
                state = next_state

        q_best = _run_sparse_pso(
            qtables,
            full_evaluator,
            dim,
            end_state,
            max_steps,
            pso_iterations,
            pso_w,
            pso_c1,
            pso_c2,
            vmin,
            vmax,
            q_clip,
            rng,
        )
        _, valid_acc, stats, selected = _qtable_score(
            q_best, full_evaluator, dim, end_state, max_steps, rng
        )
        mask = _mask_from_selected(selected, dim, q_best, xtrain, end_state)
        project_fit = _project_curve_value(mask, xtrain, xvalid, ytrain, yvalid, opts)
        if (
            valid_acc > best_valid
            or (np.isclose(valid_acc, best_valid) and stats.gain > best_gain)
            or (
                np.isclose(valid_acc, best_valid)
                and np.isclose(stats.gain, best_gain)
                and project_fit < best_project_fit
            )
        ):
            best_valid = valid_acc
            best_gain = stats.gain
            best_mask = mask.copy()
            best_q = dict(q_best)
            best_project_fit = project_fit

        curve_values.append(best_project_fit if np.isfinite(best_project_fit) else project_fit)
        qtables = [_scale_inject(q, q_best, alpha) for q in qtables]

    if best_mask is None or np.sum(best_mask) == 0:
        best_mask = _mask_from_selected([], dim, best_q, xtrain, end_state)
    nf = int(np.sum(best_mask))
    target_curve_len = int(opts.get("T", episodes))
    curve = _resize_curve(curve_values, target_curve_len)
    return {"sf": best_mask.astype(int), "c": curve, "nf": nf}
