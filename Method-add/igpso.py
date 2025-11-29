import numpy as np
from numpy.random import default_rng

from Function import Fun

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError(
        "IGPSO depends on PyTorch. Please activate the 'dl' environment or install torch."
    ) from exc


def _set_random_seeds(seed: int | None) -> default_rng:
    rng = default_rng(seed)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return rng


class FocalMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_sizes=(128, 64)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.attention = nn.Parameter(torch.ones(input_dim))
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.attention)
        return self.classifier(x * gate)

    def reset_attention(self) -> None:
        nn.init.ones_(self.attention)

    def mlp_state(self):
        return self.classifier.state_dict()

    def load_mlp_state(self, state_dict) -> None:
        self.classifier.load_state_dict(state_dict)

    def attention_vector(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.attention).clone()


def _build_wrong_labels(true_labels: np.ndarray, num_classes: int, rng: default_rng) -> np.ndarray:
    if num_classes < 2:
        return true_labels.copy()
    rand = rng.integers(0, num_classes - 1, size=true_labels.shape[0])
    wrong = rand + (rand >= true_labels)
    return wrong


def _train_stage(
    model: FocalMLP,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    min_lr: float,
    momentum: float,
    weight_decay: float,
    freeze_classifier: bool = False,
) -> tuple[np.ndarray, dict]:
    if x.size == 0:
        attn = model.attention_vector().cpu().numpy()
        return attn, model.mlp_state()

    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.int64))
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Optionally freeze the MLP when training on the negative samples so only
    # the attention vector adapts to the wrong labels. This sharpen importance
    # contrast and avoids the classifier weights drifting.
    if freeze_classifier:
        for p in model.classifier.parameters():
            p.requires_grad_(False)
        parameters = [model.attention]
    else:
        for p in model.classifier.parameters():
            p.requires_grad_(True)
        parameters = model.parameters()

    optimizer = torch.optim.SGD(
        parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=min_lr
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    attn = model.attention_vector().cpu().numpy()
    state = model.mlp_state()
    return attn, state


def generate_importance_vector(xtrain: np.ndarray, ytrain: np.ndarray, opts: dict) -> np.ndarray:
    seed = opts.get("random_seed")
    rng = _set_random_seeds(seed)

    n_samples, dim = xtrain.shape
    classes, y_encoded = np.unique(ytrain, return_inverse=True)
    num_classes = len(classes)
    pos_ratio = float(opts.get("igpso_positive_ratio", 0.7))
    pos_ratio = min(max(pos_ratio, 0.1), 0.9)

    pos_count = max(1, int(n_samples * pos_ratio))
    if pos_count > n_samples:
        pos_count = n_samples
    neg_count = max(0, n_samples - pos_count)
    if neg_count == 0 and n_samples > 1:
        neg_count = 1
        pos_count = n_samples - 1

    indices = rng.permutation(n_samples)
    pos_idx = indices[:pos_count]
    neg_idx = indices[pos_count : pos_count + neg_count]
    if neg_idx.size == 0:
        neg_idx = indices[:neg_count]

    x_pos = xtrain[pos_idx]
    y_pos = y_encoded[pos_idx]
    x_neg = xtrain[neg_idx]
    y_neg_true = y_encoded[neg_idx]
    y_neg = _build_wrong_labels(y_neg_true, num_classes, rng)

    hidden_sizes = opts.get("igpso_hidden", (128, 64))
    epochs = int(opts.get("igpso_epochs", 30))
    default_batch = min(64, len(x_pos)) if len(x_pos) > 0 else 1
    batch_size = max(1, int(opts.get("igpso_batch_size", default_batch)))
    lr = float(opts.get("igpso_lr", 0.01))
    min_lr = float(opts.get("igpso_min_lr", 0.001))
    momentum = float(opts.get("igpso_momentum", 0.9))
    weight_decay = float(opts.get("igpso_weight_decay", 1e-5))
    use_cuda = bool(opts.get("igpso_use_cuda", False))
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    model_pos = FocalMLP(dim, num_classes, hidden_sizes=hidden_sizes).to(device)
    att_pos, mlp_state = _train_stage(
        model_pos,
        x_pos,
        y_pos,
        device,
        epochs,
        batch_size,
        lr,
        min_lr,
        momentum,
        weight_decay,
        freeze_classifier=False,
    )

    model_neg = FocalMLP(dim, num_classes, hidden_sizes=hidden_sizes).to(device)
    model_neg.load_mlp_state(mlp_state)
    model_neg.reset_attention()
    att_neg, _ = _train_stage(
        model_neg,
        x_neg,
        y_neg,
        device,
        epochs,
        batch_size,
        lr,
        min_lr,
        momentum,
        weight_decay,
        freeze_classifier=True,
    )

    att_diff = att_pos - att_neg
    a_max = att_diff.max()
    a_min = att_diff.min()
    if np.isclose(a_max, a_min):
        importance = np.full(dim, 0.5, dtype=float)
    else:
        importance = ((att_diff - a_min) / (a_max - a_min)) * 0.8 + 0.1
    importance = np.clip(importance, 0.1, 0.9)
    return importance.astype(float)


def fs(xtrain: np.ndarray, xvalid: np.ndarray, ytrain: np.ndarray, yvalid: np.ndarray, opts: dict):
    ub = 1
    lb = 0
    N = int(opts["N"])
    max_iter = int(opts["T"])
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype="float")
        lb = lb * np.ones([1, dim], dtype="float")

    importance = generate_importance_vector(xtrain, ytrain, opts)

    seed = opts.get("random_seed")
    rng = _set_random_seeds(seed)

    init_random = rng.random((N, dim))
    X = (importance[None, :] > init_random).astype(int)

    fit = np.zeros([N, 1], dtype="float")
    pbest = X.copy()
    pbest_fit = np.full((N, 1), np.inf, dtype="float")
    curve = np.zeros([1, max_iter], dtype="float")

    for i in range(N):
        fit[i, 0] = Fun(xtrain, xvalid, ytrain, yvalid, X[i, :], opts)
        pbest_fit[i, 0] = fit[i, 0]
    best_idx = int(np.argmin(pbest_fit[:, 0]))
    gbest = pbest[best_idx, :].copy()
    gbest_fit = pbest_fit[best_idx, 0]

    for t in range(max_iter):
        for i in range(N):
            rand_vec = rng.random(dim)
            pb_term = 0.5 * importance * np.abs(pbest[i, :] - X[i, :])
            gb_term = 0.5 * importance * np.abs(gbest - X[i, :])
            flip_prob = (1.0 - importance) * rand_vec + pb_term + gb_term
            flip_prob = np.clip(flip_prob, 0.0, 1.0)
            flip_mask = rng.random(dim) < flip_prob
            if np.any(flip_mask):
                X[i, flip_mask] = 1 - X[i, flip_mask]

            fit_i = Fun(xtrain, xvalid, ytrain, yvalid, X[i, :], opts)
            if fit_i < pbest_fit[i, 0]:
                pbest[i, :] = X[i, :].copy()
                pbest_fit[i, 0] = fit_i
                if fit_i < gbest_fit:
                    gbest = X[i, :].copy()
                    gbest_fit = fit_i

        curve[0, t] = gbest_fit

    sf = gbest.astype(int).reshape(dim)
    if np.sum(sf) == 0:
        sf[np.argmax(importance)] = 1
    num_feat = int(np.sum(sf))
    igpso_data = {"sf": sf, "c": curve, "nf": num_feat}
    return igpso_data
