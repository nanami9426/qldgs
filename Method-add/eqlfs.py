import math
import os
import time
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
    clean = {}
    for state, row in q.items():
        clean_row = {
            int(action): float(value)
            for action, value in row.items()
            if abs(value) > _ZERO_TOL
        }
        if clean_row:
            clean[int(state)] = clean_row
    return clean


def _copy_qtable(q):
    return {int(state): dict(row) for state, row in q.items()}


def _iter_keys(q):
    for state, row in q.items():
        for action in row:
            yield int(state), int(action)


def _q_get(q, state, action):
    return float(q.get(int(state), {}).get(int(action), 0.0))


def _q_set(q, state, action, value, q_clip):
    state = int(state)
    action = int(action)
    value = float(np.clip(value, -q_clip, q_clip))
    if abs(value) <= _ZERO_TOL:
        row = q.get(state)
        if row is not None:
            row.pop(action, None)
            if not row:
                q.pop(state, None)
        return
    q.setdefault(state, {})[action] = value


def _is_action_available(action, selected_set, dim, end_state):
    return action == end_state or (0 <= action < dim and action not in selected_set)


def _random_available_action(dim, selected_set, end_state, rng):
    if len(selected_set) >= dim:
        return end_state
    for _ in range(32):
        action = int(rng.integers(0, dim + 1))
        if _is_action_available(action, selected_set, dim, end_state):
            return action
    while True:
        action = int(rng.integers(0, dim + 1))
        if _is_action_available(action, selected_set, dim, end_state):
            return action


def _known_available_best(q, state, selected_set, dim, end_state):
    best_value = -np.inf
    best_actions = []
    for action, value in q.get(int(state), {}).items():
        action = int(action)
        if not _is_action_available(action, selected_set, dim, end_state):
            continue
        value = float(value)
        if value > best_value + _ZERO_TOL:
            best_value = value
            best_actions = [action]
        elif np.isclose(value, best_value):
            best_actions.append(action)
    return best_value, best_actions


def _max_available_q(q, state, selected_set, dim, end_state):
    best_value, _ = _known_available_best(q, state, selected_set, dim, end_state)
    return float(max(0.0, best_value))


def _sample_next_action(q, state, selected_set, dim, end_state, epsilon, rng):
    if rng.random() < epsilon:
        return _random_available_action(dim, selected_set, end_state, rng)

    best_value, best_actions = _known_available_best(
        q, state, selected_set, dim, end_state
    )
    if best_actions and best_value > 0:
        return int(rng.choice(best_actions))
    return _random_available_action(dim, selected_set, end_state, rng)


def _greedy_subset(q, dim, end_state, max_steps, rng):
    selected = []
    selected_set = set()
    state = end_state
    for _ in range(max_steps):
        best_value, best_actions = _known_available_best(
            q, state, selected_set, dim, end_state
        )
        if best_actions and best_value > 0:
            action = int(rng.choice(best_actions))
        else:
            action = _random_available_action(dim, selected_set, end_state, rng)
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
    return [_copy_qtable(q) for q in qtables]


def _scale_inject(q, q_best, alpha):
    keys = set(_iter_keys(q)) | set(_iter_keys(q_best))
    mixed = {}
    for state, action in keys:
        value = (1.0 - alpha) * _q_get(q_best, state, action) + alpha * _q_get(
            q, state, action
        )
        _q_set(mixed, state, action, value, q_clip=np.inf)
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
    gbest = _copy_qtable(qtables[0])
    gbest_score = -np.inf
    gbest_valid = -np.inf

    for i, q in enumerate(qtables):
        score, valid_acc, _, _ = _qtable_score(q, evaluator, dim, end_state, max_steps, rng)
        pbest_scores[i] = score
        if (score > gbest_score) or (np.isclose(score, gbest_score) and valid_acc > gbest_valid):
            gbest = _copy_qtable(q)
            gbest_score = score
            gbest_valid = valid_acc

    for _ in range(max(1, pso_iterations)):
        for i in range(particle_count):
            keys = (
                set(_iter_keys(qtables[i]))
                | set(_iter_keys(velocities[i]))
                | set(_iter_keys(pbest[i]))
                | set(_iter_keys(gbest))
            )
            next_q = {}
            next_v = {}
            for state, action in keys:
                x = _q_get(qtables[i], state, action)
                vel = _q_get(velocities[i], state, action)
                r1 = rng.random()
                r2 = rng.random()
                vel = (
                    w * vel
                    + c1 * r1 * (_q_get(pbest[i], state, action) - x)
                    + c2 * r2 * (_q_get(gbest, state, action) - x)
                )
                vel = float(np.clip(vel, vmin, vmax))
                value = float(np.clip(x + vel, -q_clip, q_clip))
                _q_set(next_q, state, action, value, q_clip)
                _q_set(next_v, state, action, vel, q_clip=np.inf)
            qtables[i] = next_q
            velocities[i] = next_v

            score, valid_acc, _, _ = _qtable_score(
                qtables[i], evaluator, dim, end_state, max_steps, rng
            )
            if score > pbest_scores[i]:
                pbest_scores[i] = score
                pbest[i] = _copy_qtable(qtables[i])
            if (score > gbest_score) or (
                np.isclose(score, gbest_score) and valid_acc > gbest_valid
            ):
                gbest = _copy_qtable(qtables[i])
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
    row = q_best.get(int(end_state), {})
    candidates = [
        (int(action), float(value))
        for action, value in row.items()
        if 0 <= int(action) < dim
    ]
    if candidates:
        return max(candidates, key=lambda item: item[1])[0]
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


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


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
    verbose = _as_bool(opts.get("eqlfs_verbose", True))
    progress_interval = int(opts.get("eqlfs_progress_interval", max(1, episodes // 10)))
    progress_interval = max(0, progress_interval)
    data_name = opts.get("data_name", "dataset")
    method_name = opts.get("method_name", "eqlfs")
    run_index = opts.get("run_index", "?")
    run_total = opts.get("run_total", "?")
    progress_prefix = (
        f"[{data_name}][{method_name}][run {run_index}/{run_total}]"
        f"[pid {os.getpid()}]"
    )

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
    started_at = time.time()
    if verbose:
        print(
            f"{progress_prefix} EQLFS start: dim={dim}, agents={agents}, "
            f"episodes={episodes}, max_steps={max_steps}",
            flush=True,
        )

    for episode in range(episodes):
        sample_idx = _episode_indices(ytrain, episode_ratio, rng)
        episode_evaluator = DecisionTreeSubsetEvaluator(
            xtrain[sample_idx],
            xvalid,
            ytrain[sample_idx],
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
                    q, state, selected_set, dim, end_state, epsilon, rng
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
                td_target = reward + gamma * _max_available_q(
                    q, next_state, selected_set, dim, end_state
                )
                old_value = _q_get(q, state, action)
                new_value = old_value + lr * (td_target - old_value)
                _q_set(q, state, action, new_value, q_clip)
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
            best_q = _copy_qtable(q_best)
            best_project_fit = project_fit

        curve_values.append(best_project_fit if np.isfinite(best_project_fit) else project_fit)
        qtables = [_scale_inject(q, q_best, alpha) for q in qtables]
        if verbose and progress_interval > 0:
            episode_no = episode + 1
            if episode_no == 1 or episode_no == episodes or episode_no % progress_interval == 0:
                elapsed = time.time() - started_at
                best_nf = int(np.sum(best_mask)) if best_mask is not None else 0
                print(
                    f"{progress_prefix} episode {episode_no}/{episodes} "
                    f"elapsed={elapsed:.1f}s best_fit={best_project_fit:.6f} "
                    f"best_valid={best_valid:.4f} best_nf={best_nf}",
                    flush=True,
                )

    if best_mask is None or np.sum(best_mask) == 0:
        best_mask = _mask_from_selected([], dim, best_q, xtrain, end_state)
    nf = int(np.sum(best_mask))
    target_curve_len = int(opts.get("T", episodes))
    curve = _resize_curve(curve_values, target_curve_len)
    if verbose:
        print(
            f"{progress_prefix} EQLFS done: elapsed={time.time() - started_at:.1f}s "
            f"nf={nf} best_fit={best_project_fit:.6f}",
            flush=True,
        )
    return {"sf": best_mask.astype(int), "c": curve, "nf": nf}
