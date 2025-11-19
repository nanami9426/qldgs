import argparse
import importlib.util
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from CCM import calculate_metrics
from Function import classifier_method

DEFAULT_MODES = {
    "baseline_pso": {"ql_mode": "baseline"},
    "pso_dim": {"ql_mode": "dim_only"},
    "pso_dim_random": {"ql_mode": "random_policy"},
    "full_qldgs": {"ql_mode": "full"},
}


def load_qldgs_module():
    module_path = Path(__file__).resolve().parent / "Method-combination" / "QLDGS-PSO-Elite.py"
    spec = importlib.util.spec_from_file_location("qldgs_pso_elite", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_datasets(dataset_names):
    dataset_dir = Path("Dataset")
    if dataset_names:
        return dataset_names
    return sorted([p.stem for p in dataset_dir.glob("*.csv")])


def load_dataset(name):
    path = Path("Dataset") / f"{name}.csv"
    data = pd.read_csv(path)
    X = data.iloc[:, :-1].values.copy()
    Y = data.iloc[:, -1].values.copy()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.nan_to_num(X)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    return X, Y


def evaluate_configuration(qldgs, X, Y, opts, runs, test_size, valid_size):
    dim = X.shape[1]
    results = []
    for run in range(runs):
        X_train_val, X_test, Y_train_val, Y_test = train_test_split(
            X, Y, test_size=test_size, stratify=Y, random_state=run
        )
        X_train, X_valid, Y_train, Y_valid = train_test_split(
            X_train_val, Y_train_val, test_size=valid_size, stratify=Y_train_val, random_state=run
        )
        local_opts = opts.copy()
        local_opts["random_seed"] = run
        start = time.time()
        FS = qldgs.fs(X_train.copy(), X_valid.copy(), Y_train.copy(), Y_valid.copy(), local_opts)
        elapsed = time.time() - start
        selected = FS["sf"] == 1
        if not np.any(selected):
            selected = np.ones(dim, dtype=bool)
        valid_pred = classifier_method(
            X_train[:, selected], X_valid[:, selected], Y_train, Y_valid, local_opts
        )
        test_pred = classifier_method(
            X_train[:, selected], X_test[:, selected], Y_train, Y_test, local_opts
        )
        valid_metrics = calculate_metrics(Y_valid, valid_pred, "all")
        test_metrics = calculate_metrics(Y_test, test_pred, "all")
        results.append(
            {
                "nf": FS["nf"],
                "time": elapsed,
                "acc_valid": valid_metrics["Accuracy"],
                "acc_test": test_metrics["Accuracy"],
                "f1_valid": valid_metrics["F1-Score (Macro)"],
                "f1_test": test_metrics["F1-Score (Macro)"],
                "auc_valid": valid_metrics["ROC AUC (Macro)"],
                "auc_test": test_metrics["ROC AUC (Macro)"],
            }
        )
    return results


def run(args):
    datasets = discover_datasets(args.datasets)
    if not datasets:
        raise RuntimeError("No datasets found for ablation study")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    if not 0 < args.valid_size < 1:
        raise ValueError("--valid-size must be between 0 and 1")
    qldgs = load_qldgs_module()
    selected_modes = args.modes or list(DEFAULT_MODES.keys())
    records = []
    for dataset in datasets:
        X, Y = load_dataset(dataset)
        for mode in selected_modes:
            if mode not in DEFAULT_MODES:
                raise ValueError(f"Unknown mode '{mode}'. Available: {list(DEFAULT_MODES.keys())}")
            mode_cfg = DEFAULT_MODES[mode]
            local_opts = {
                "N": args.population,
                "T": args.iterations,
                "ql_lr": args.ql_lr,
                "ql_gamma": args.ql_gamma,
                "interval_num": args.interval_num,
                "interval_iterations": args.interval_iterations,
                "phi": args.phi,
                "cluster": ",",
                "classify": args.classifier,
                "split": args.split,
                "func": args.func,
                "knn_para": args.knn_k,
            }
            local_opts.update(mode_cfg)
            print(f"{dataset} -> {mode}")
            mode_results = evaluate_configuration(
                qldgs, X, Y, local_opts, args.runs, args.test_size, args.valid_size
            )
            for run_idx, metrics in enumerate(mode_results):
                record = {
                    "dataset": dataset,
                    "mode": mode,
                    "run": run_idx,
                    "ql_mode": mode_cfg.get("ql_mode"),
                    "nf": metrics["nf"],
                    "time": metrics["time"],
                    "acc_valid": metrics["acc_valid"],
                    "acc_test": metrics["acc_test"],
                    "f1_valid": metrics["f1_valid"],
                    "f1_test": metrics["f1_test"],
                    "auc_valid": metrics["auc_valid"],
                    "auc_test": metrics["auc_test"],
                }
                records.append(record)
    df = pd.DataFrame(records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    summary = (
        df.groupby(["dataset", "mode"])
        .agg(
            acc_test_mean=("acc_test", "mean"),
            acc_test_std=("acc_test", "std"),
            f1_test_mean=("f1_test", "mean"),
            f1_test_std=("f1_test", "std"),
            auc_test_mean=("auc_test", "mean"),
            auc_test_std=("auc_test", "std"),
            nf_mean=("nf", "mean"),
            nf_std=("nf", "std"),
            time_mean=("time", "mean"),
        )
        .reset_index()
    )
    summary_path = output.with_name(f"{output.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Ablation raw logs saved to {output}")
    print(f"Ablation summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="QLDGS-PSO-Elite ablation runner")
    parser.add_argument("--datasets", nargs="*", help="Dataset names without .csv extension")
    parser.add_argument("--runs", type=int, default=5, help="Independent runs per mode")
    parser.add_argument("--population", type=int, default=20, help="Population size N")
    parser.add_argument("--iterations", type=int, default=500, help="Iterations per run T")
    parser.add_argument("--interval-num", type=int, default=20, help="Number of intervals L")
    parser.add_argument(
        "--interval-iterations", type=int, default=20, help="Iterations per interval M"
    )
    parser.add_argument("--ql-lr", type=float, default=0.2, help="Q-learning learning rate lr")
    parser.add_argument("--ql-gamma", type=float, default=0.5, help="Q-learning discount γ")
    parser.add_argument("--phi", type=float, default=0.99, help="Fitness weighting φ")
    parser.add_argument("--classifier", type=str, default="knn", help="Classifier choice")
    parser.add_argument("--knn-k", type=int, default=3, help="k for KNN classifier")
    parser.add_argument("--split", type=int, default=0, help="Split strategy used in Fun")
    parser.add_argument("--func", type=int, default=0, help="Objective variant")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction reserved for held-out test data"
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.25,
        help="Fraction of remaining data used for validation",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        choices=list(DEFAULT_MODES.keys()),
        help="Subset of ablation modes to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Result/ablation_experiments.csv",
        help="CSV file for ablation logs",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
