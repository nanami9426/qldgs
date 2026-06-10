from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
from numpy.random import Generator, default_rng
from sklearn.metrics import accuracy_score

from Function import Fun, classifier_method

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError(
        "RLFS depends on PyTorch. Please activate an environment with torch installed."
    ) from exc


_ZERO_TOL = 1e-12


def _set_random_seeds(seed: int | None) -> Generator:
    rng = default_rng(seed)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return rng


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _feature_relevance_scores(xtrain, ytrain) -> np.ndarray:
    xtrain = np.nan_to_num(np.asarray(xtrain, dtype=float))
    ytrain = np.asarray(ytrain).reshape(-1)
    dim = xtrain.shape[1]
    scores = np.zeros(dim, dtype=float)
    classes = np.unique(ytrain)
    if len(classes) <= 1 or dim == 0:
        return scores

    overall_mean = np.mean(xtrain, axis=0)
    between = np.zeros(dim, dtype=float)
    within = np.zeros(dim, dtype=float)
    for cls in classes:
        cls_x = xtrain[ytrain == cls]
        if cls_x.size == 0:
            continue
        cls_mean = np.mean(cls_x, axis=0)
        between += cls_x.shape[0] * (cls_mean - overall_mean) ** 2
        if cls_x.shape[0] > 1:
            within += np.var(cls_x, axis=0) * (cls_x.shape[0] - 1)

    scores = between / (within + _ZERO_TOL)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    max_score = scores.max(initial=0.0)
    if max_score > 0:
        scores = scores / max_score
    return scores


def _candidate_feature_indices(xtrain, ytrain, candidate_count, rng: Generator):
    dim = xtrain.shape[1]
    scores = _feature_relevance_scores(xtrain, ytrain)
    if candidate_count >= dim:
        return np.arange(dim, dtype=int), scores

    ranked = np.argsort(-scores, kind="stable")
    candidate_count = max(1, min(dim, int(candidate_count)))
    selected = ranked[:candidate_count]

    random_ratio = 0.1
    random_count = int(round(candidate_count * random_ratio))
    if random_count > 0 and candidate_count < dim:
        top_count = max(1, candidate_count - random_count)
        top = ranked[:top_count]
        remaining = ranked[top_count:]
        random_tail = rng.choice(
            remaining, size=min(random_count, len(remaining)), replace=False
        )
        selected = np.concatenate((top, random_tail)).astype(int)
        if len(selected) < candidate_count:
            selected_set = set(int(i) for i in selected)
            fill = [idx for idx in ranked if int(idx) not in selected_set]
            selected = np.concatenate(
                (selected, np.asarray(fill[: candidate_count - len(selected)]))
            )

    return np.asarray(selected[:candidate_count], dtype=int), scores


def _balanced_under_sample(X, y, sample_size: int, rng: Generator):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) <= 1:
        take = min(len(y), max(1, int(sample_size)))
        idx = rng.choice(len(y), size=take, replace=False)
        return X[idx], y[idx]

    per_class = max(1, int(sample_size) // len(classes))
    per_class = min(per_class, int(np.min(counts)))
    indices = []
    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        indices.append(rng.choice(cls_idx, size=per_class, replace=False))
    idx = np.concatenate(indices)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _majority_accuracy(y) -> float:
    y = np.asarray(y).reshape(-1)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    return float(np.max(counts) / y.size)


def _loo_1nn_accuracy(X, y) -> float:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2 or X.shape[0] <= 1 or X.shape[1] == 0:
        return _majority_accuracy(y)

    sq_norm = np.sum(X * X, axis=1)
    dist = sq_norm[:, None] + sq_norm[None, :] - 2.0 * X @ X.T
    np.maximum(dist, 0.0, out=dist)
    np.fill_diagonal(dist, np.inf)
    pred = y[np.argmin(dist, axis=1)]
    return float(np.mean(pred == y))


class IncrementalLOO1NN:
    def __init__(self, X, y):
        self.X = np.nan_to_num(np.asarray(X, dtype=float))
        self.y = np.asarray(y).reshape(-1)
        n = self.X.shape[0]
        self.dist = np.zeros((n, n), dtype=float)
        np.fill_diagonal(self.dist, np.inf)

    def empty_accuracy(self) -> float:
        return _majority_accuracy(self.y)

    def add_feature(self, feature_index: int) -> float:
        col = self.X[:, int(feature_index)]
        diff = col[:, None] - col[None, :]
        self.dist += diff * diff
        np.fill_diagonal(self.dist, np.inf)
        pred = self.y[np.argmin(self.dist, axis=1)]
        return float(np.mean(pred == self.y))


class FeatureAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.ReLU())
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


class CompleteGraphStateProjector:
    def __init__(self, state_dim: int, hidden_dim: int, seed: int | None):
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.seed = 0 if seed is None else int(seed)
        self.weights: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _weights_for(self, input_dim: int):
        if input_dim not in self.weights:
            rng = default_rng(self.seed + input_dim * 7919)
            w1 = rng.normal(
                0.0, 1.0 / max(1.0, np.sqrt(input_dim)), size=(input_dim, self.hidden_dim)
            )
            w2 = rng.normal(
                0.0,
                1.0 / max(1.0, np.sqrt(self.hidden_dim)),
                size=(self.hidden_dim, self.state_dim),
            )
            self.weights[input_dim] = (w1.astype(np.float32), w2.astype(np.float32))
        return self.weights[input_dim]

    def project(self, latent: np.ndarray) -> np.ndarray:
        if latent.size == 0:
            return np.zeros(self.state_dim, dtype=np.float32)
        pooled = np.mean(latent, axis=0).astype(np.float32)
        w1, w2 = self._weights_for(pooled.size)
        h = np.maximum(pooled @ w1, 0.0)
        out = np.maximum(h @ w2, 0.0)
        norm = np.linalg.norm(out)
        if norm > 0:
            out = out / norm
        return out.astype(np.float32)


def _state_representation(
    X_balanced,
    latent_dim: int,
    state_projector: CompleteGraphStateProjector,
    epochs: int,
    lr: float,
    device: torch.device,
) -> np.ndarray:
    X_balanced = np.nan_to_num(np.asarray(X_balanced, dtype=np.float32))
    if X_balanced.ndim != 2 or X_balanced.shape[0] == 0 or X_balanced.shape[1] == 0:
        return np.zeros(state_projector.state_dim, dtype=np.float32)

    feature_columns = X_balanced.T
    input_dim = feature_columns.shape[1]
    latent_dim = max(1, min(int(latent_dim), input_dim))
    x_tensor = torch.from_numpy(feature_columns).to(device)

    model = FeatureAutoEncoder(input_dim, latent_dim).to(device)
    if epochs > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        model.train()
        for _ in range(int(epochs)):
            optimizer.zero_grad()
            recon = model(x_tensor)
            loss = criterion(recon, x_tensor)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        latent = model.encode(x_tensor).detach().cpu().numpy()
    return state_projector.project(latent)


class DecisionNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(64, 32)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 2),
        )

    def forward(self, state):
        return self.net(state)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data: Deque[Transition] = deque(maxlen=int(capacity))

    def append(self, transition: Transition):
        self.data.append(transition)

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size: int, rng: Generator):
        idx = rng.choice(len(self.data), size=int(batch_size), replace=False)
        return [self.data[int(i)] for i in idx]


def _make_state(dataset_state, step_index: int, dim: int, selected_mask) -> np.ndarray:
    if dim <= 1:
        step_value = 0.0
    else:
        step_value = step_index / (dim - 1)
    return np.concatenate(
        (
            np.asarray(dataset_state, dtype=np.float32),
            np.asarray([step_value], dtype=np.float32),
            np.asarray(selected_mask, dtype=np.float32),
        )
    ).astype(np.float32)


def _choose_action(
    policy_net: DecisionNetwork,
    state: np.ndarray,
    epsilon: float,
    rng: Generator,
    device: torch.device,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, 2))
    with torch.no_grad():
        q_values = policy_net(torch.from_numpy(state).to(device).unsqueeze(0))
    return int(torch.argmax(q_values, dim=1).item())


def _optimize_dqn(
    policy_net: DecisionNetwork,
    target_net: DecisionNetwork,
    optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    tau: float,
    rng: Generator,
    device: torch.device,
) -> float | None:
    if len(replay) < batch_size:
        return None

    batch = replay.sample(batch_size, rng)
    states = torch.from_numpy(np.stack([b.state for b in batch])).to(device)
    actions = torch.tensor([b.action for b in batch], dtype=torch.long, device=device).view(-1, 1)
    rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=device)
    next_states = torch.from_numpy(np.stack([b.next_state for b in batch])).to(device)
    done = torch.tensor([b.done for b in batch], dtype=torch.float32, device=device)

    q_values = policy_net(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1).values
        target = rewards + gamma * (1.0 - done) * next_q
    loss = nn.functional.mse_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
    optimizer.step()

    with torch.no_grad():
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(policy_param.data, alpha=tau)
    return float(loss.item())


def _validation_accuracy(mask, xtrain, xvalid, ytrain, yvalid, opts) -> float:
    mask = np.asarray(mask).reshape(-1).astype(int)
    if mask.sum() == 0:
        return 0.0
    try:
        pred = classifier_method(
            xtrain[:, mask == 1], xvalid[:, mask == 1], ytrain, yvalid, opts
        )
        return float(accuracy_score(np.asarray(yvalid).reshape(-1), pred))
    except Exception:
        return 0.0


def _project_cost(mask, xtrain, xvalid, ytrain, yvalid, opts) -> float:
    mask = np.asarray(mask).reshape(-1).astype(int)
    if mask.sum() == 0:
        return 1.0
    try:
        return float(Fun(xtrain, xvalid, ytrain, yvalid, mask, opts))
    except Exception:
        return 1.0


def _resize_curve(values, target_len: int) -> np.ndarray:
    target_len = max(1, int(target_len))
    if not values:
        return np.ones(target_len, dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if len(arr) == target_len:
        return arr
    if len(arr) < target_len:
        out = np.empty(target_len, dtype=float)
        out[: len(arr)] = arr
        out[len(arr) :] = arr[-1]
        return out
    idx = np.round(np.linspace(0, len(arr) - 1, target_len)).astype(int)
    return arr[idx]


def _greedy_feature_scan(
    policy_net,
    dataset_state,
    X_balanced,
    y_balanced,
    epsilon,
    rng,
    device,
):
    dim = X_balanced.shape[1]
    selected = np.zeros(dim, dtype=np.float32)
    evaluator = IncrementalLOO1NN(X_balanced, y_balanced)
    current_acc = evaluator.empty_accuracy()
    for j in range(dim):
        state = _make_state(dataset_state, j, dim, selected)
        action = _choose_action(policy_net, state, epsilon, rng, device)
        if action == 1:
            selected[j] = 1.0
            current_acc = evaluator.add_feature(j)
    return selected.astype(int), float(current_acc)


def _forward_selection_from_weights(
    weights,
    candidate_idx,
    xtrain,
    xvalid,
    ytrain,
    yvalid,
    opts,
):
    original_dim = xtrain.shape[1]
    order = np.argsort(-np.asarray(weights, dtype=float), kind="stable")
    best_mask = np.zeros(original_dim, dtype=int)
    best_acc = -np.inf
    working_mask = np.zeros(original_dim, dtype=int)
    patience = int(opts.get("rlfs_forward_patience", 0))
    stale_rounds = 0

    for internal_idx in order:
        if weights[internal_idx] <= 0 and best_mask.sum() > 0:
            break
        working_mask[int(candidate_idx[internal_idx])] = 1
        acc = _validation_accuracy(working_mask, xtrain, xvalid, ytrain, yvalid, opts)
        if acc > best_acc + _ZERO_TOL:
            best_acc = acc
            best_mask = working_mask.copy()
            stale_rounds = 0
        else:
            stale_rounds += 1
            if patience > 0 and stale_rounds >= patience and best_mask.sum() > 0:
                break

    if best_mask.sum() == 0:
        best_internal = int(order[0]) if len(order) else 0
        best_mask[int(candidate_idx[best_internal])] = 1
    return best_mask


def fs(xtrain, xvalid, ytrain, yvalid, opts=None):
    """
    Reinforcement learning-based feature selection for imbalanced data (RLFS).

    This implementation follows the method in:
    "A Deep Reinforcement Learning-Based Feature Selection Method for Invasive
    Disease Event Prediction Using Imbalanced Follow-Up Data".

    The paper settings can be approached with:
        rlfs_episodes=4000, rlfs_weight_episodes=50, rlfs_sample_size=200

    For high-dimensional benchmark datasets in this repository, a candidate
    feature prefilter is enabled by default through rlfs_candidate_features.
    """
    if opts is None:
        opts = {}
    opts = dict(opts)

    seed = opts.get("random_seed")
    rng = _set_random_seeds(seed)
    xtrain = np.nan_to_num(np.asarray(xtrain, dtype=float))
    xvalid = np.nan_to_num(np.asarray(xvalid, dtype=float))
    ytrain = np.asarray(ytrain).reshape(-1)
    yvalid = np.asarray(yvalid).reshape(-1)
    original_dim = int(xtrain.shape[1])
    target_curve_len = int(opts.get("T", 100))
    if original_dim == 0:
        return {"sf": np.zeros(0, dtype=int), "c": np.ones(target_curve_len), "nf": 0}

    default_candidates = min(original_dim, int(opts.get("rlfs_default_candidate_features", 256)))
    candidate_count = int(opts.get("rlfs_candidate_features", default_candidates))
    candidate_idx, relevance = _candidate_feature_indices(xtrain, ytrain, candidate_count, rng)
    xtrain_c = xtrain[:, candidate_idx]
    xvalid_c = xvalid[:, candidate_idx]
    dim = int(xtrain_c.shape[1])

    episodes = max(1, int(opts.get("rlfs_episodes", min(target_curve_len, 120))))
    weight_episodes = max(1, int(opts.get("rlfs_weight_episodes", min(50, max(10, episodes // 5)))))
    sample_size = max(2, int(opts.get("rlfs_sample_size", 200)))
    latent_dim = int(opts.get("rlfs_ae_latent_dim", 128))
    state_dim = int(opts.get("rlfs_state_dim", 64))
    gcn_hidden = int(opts.get("rlfs_gcn_hidden", 128))
    state_epochs = max(0, int(opts.get("rlfs_state_epochs", 5)))
    state_lr = float(opts.get("rlfs_state_lr", 0.01))
    gamma = float(opts.get("rlfs_gamma", 0.99))
    epsilon0 = float(opts.get("rlfs_epsilon", 0.5))
    epsilon_min = float(opts.get("rlfs_epsilon_min", 0.05))
    lr0 = float(opts.get("rlfs_lr", 0.025))
    min_lr = float(opts.get("rlfs_min_lr", 0.001))
    tau = float(opts.get("rlfs_tau", 0.01))
    batch_size = max(1, int(opts.get("rlfs_batch_size", 64)))
    memory_size = max(batch_size, int(opts.get("rlfs_memory_size", 20000)))
    train_interval = max(1, int(opts.get("rlfs_train_interval", 64)))
    hidden_sizes = opts.get("rlfs_hidden", (64, 32))
    use_cuda = _as_bool(opts.get("rlfs_use_cuda", False))
    verbose = _as_bool(opts.get("rlfs_verbose", False))
    progress_interval = max(0, int(opts.get("rlfs_progress_interval", 25)))

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    input_dim = state_dim + 1 + dim
    policy_net = DecisionNetwork(input_dim, hidden_sizes=hidden_sizes).to(device)
    target_net = DecisionNetwork(input_dim, hidden_sizes=hidden_sizes).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr0)
    replay = ReplayBuffer(memory_size)
    state_projector = CompleteGraphStateProjector(state_dim, gcn_hidden, seed)

    best_mask = np.zeros(original_dim, dtype=int)
    best_cost = np.inf
    curve_values = []
    global_step = 0

    X_env = np.vstack((xtrain_c, xvalid_c))
    y_env = np.concatenate((ytrain, yvalid))
    opts_eval = dict(opts)
    opts_eval["dim"] = original_dim

    for episode in range(episodes):
        frac = episode / max(1, episodes - 1)
        epsilon = epsilon0 + (epsilon_min - epsilon0) * frac
        lr = lr0 + (min_lr - lr0) * frac
        for group in optimizer.param_groups:
            group["lr"] = lr

        X_bal, y_bal = _balanced_under_sample(X_env, y_env, sample_size, rng)
        dataset_state = _state_representation(
            X_bal, latent_dim, state_projector, state_epochs, state_lr, device
        )
        selected = np.zeros(dim, dtype=np.float32)
        evaluator = IncrementalLOO1NN(X_bal, y_bal)
        current_acc = evaluator.empty_accuracy()

        for j in range(dim):
            state = _make_state(dataset_state, j, dim, selected)
            action = _choose_action(policy_net, state, epsilon, rng, device)
            reward = 0.0
            if action == 1:
                selected[j] = 1.0
                next_acc = evaluator.add_feature(j)
                reward = next_acc - current_acc
                current_acc = next_acc

            next_state = _make_state(dataset_state, min(j + 1, dim - 1), dim, selected)
            done = j == dim - 1
            replay.append(Transition(state, action, reward, next_state, done))
            global_step += 1
            if global_step % train_interval == 0:
                _optimize_dqn(
                    policy_net,
                    target_net,
                    optimizer,
                    replay,
                    batch_size,
                    gamma,
                    tau,
                    rng,
                    device,
                )

        original_mask = np.zeros(original_dim, dtype=int)
        internal_selected = np.flatnonzero(selected.astype(int))
        if internal_selected.size == 0:
            internal_selected = np.asarray([int(np.argmax(relevance[candidate_idx]))])
        original_mask[candidate_idx[internal_selected]] = 1
        cost = _project_cost(original_mask, xtrain, xvalid, ytrain, yvalid, opts_eval)
        if cost < best_cost:
            best_cost = cost
            best_mask = original_mask.copy()
        curve_values.append(best_cost)

        if verbose and progress_interval > 0:
            episode_no = episode + 1
            if episode_no == 1 or episode_no == episodes or episode_no % progress_interval == 0:
                print(
                    f"[RLFS] episode {episode_no}/{episodes} "
                    f"epsilon={epsilon:.3f} best_fit={best_cost:.6f} "
                    f"best_nf={int(best_mask.sum())}",
                    flush=True,
                )

    weights = np.zeros(dim, dtype=float)
    policy_net.eval()
    for _ in range(weight_episodes):
        X_bal, y_bal = _balanced_under_sample(X_env, y_env, sample_size, rng)
        dataset_state = _state_representation(
            X_bal, latent_dim, state_projector, state_epochs, state_lr, device
        )
        selected, selected_acc = _greedy_feature_scan(
            policy_net, dataset_state, X_bal, y_bal, 0.0, rng, device
        )
        baseline = _loo_1nn_accuracy(X_bal, y_bal)
        if baseline <= _ZERO_TOL:
            continue
        weights += selected * ((selected_acc - baseline) / baseline)

    if np.max(weights, initial=0.0) <= 0.0 and best_mask.sum() > 0:
        final_mask = best_mask.copy()
    else:
        final_mask = _forward_selection_from_weights(
            weights, candidate_idx, xtrain, xvalid, ytrain, yvalid, opts
        )

    final_cost = _project_cost(final_mask, xtrain, xvalid, ytrain, yvalid, opts_eval)
    if final_cost < best_cost:
        best_cost = final_cost
        best_mask = final_mask.copy()
    else:
        final_mask = best_mask.copy()
    curve_values.append(best_cost)

    if final_mask.sum() == 0:
        fallback = int(candidate_idx[int(np.argmax(relevance[candidate_idx]))])
        final_mask[fallback] = 1

    curve = _resize_curve(curve_values, target_curve_len)
    return {"sf": final_mask.astype(int), "c": curve, "nf": int(final_mask.sum())}
