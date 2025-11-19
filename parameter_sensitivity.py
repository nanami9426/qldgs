import argparse
import itertools
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from CCM import calculate_metrics
from Function import classifier_method


def load_qldgs_module():
    module_path = Path(__file__).resolve().parent / "Method-combination" / "QLDGS-PSO-Elite.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot locate QLDGS-PSO-Elite.py at {module_path}")
    spec = importlib.util.spec_from_file_location("qldgs_pso_elite", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_numeric_list(text, cast_func=float):
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(cast_func(item))
    return values


def discover_datasets(dataset_names):
    dataset_dir = Path("Dataset")
    if dataset_names:
        return dataset_names
    return sorted([p.stem for p in dataset_dir.glob("*.csv")])


def load_dataset(name):
    dataset_path = Path("Dataset") / f"{name}.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_path} not found")
    data = pd.read_csv(dataset_path)
    X = data.iloc[:, :-1].values.copy()
    Y = data.iloc[:, -1].values.copy()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.nan_to_num(X)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    return X, Y


def stratified_split(X, Y, test_size, random_state):
    """Wrapper around train_test_split that only stratifies when possible."""
    stratify = Y if len(np.unique(Y)) > 1 else None
    return train_test_split(
        X, Y, test_size=test_size, stratify=stratify, random_state=random_state
    )


def create_data_splits(X, Y, run, split_mode, test_size, valid_size):
    """Prepare train/valid/test splits for a given run."""
    if split_mode == "train_val_test":
        X_train_val, X_test, Y_train_val, Y_test = stratified_split(
            X, Y, test_size=test_size, random_state=run
        )
        X_train, X_valid, Y_train, Y_valid = stratified_split(
            X_train_val, Y_train_val, test_size=valid_size, random_state=run
        )
        return {
            "fs_train": (X_train, Y_train),
            "fs_valid": (X_valid, Y_valid),
            "train_eval": (X_train, Y_train),
            "valid_eval": (X_valid, Y_valid),
            "test": (X_test, Y_test),
            "has_valid_eval": True,
        }
    if split_mode == "main_exvute":
        X_train, X_test, Y_train, Y_test = stratified_split(
            X, Y, test_size=test_size, random_state=run
        )
        return {
            "fs_train": (X_train, Y_train),
            "fs_valid": (X_test, Y_test),
            "train_eval": (X_train, Y_train),
            "valid_eval": (X_test, Y_test),
            "test": (X_test, Y_test),
            "has_valid_eval": True,
        }
    # train/test mode still keeps an internal validation slice for FS fitness
    X_train_full, X_test, Y_train_full, Y_test = stratified_split(
        X, Y, test_size=test_size, random_state=run
    )
    X_train_fs, X_valid_fs, Y_train_fs, Y_valid_fs = stratified_split(
        X_train_full, Y_train_full, test_size=valid_size, random_state=run
    )
    return {
        "fs_train": (X_train_fs, Y_train_fs),
        "fs_valid": (X_valid_fs, Y_valid_fs),
        "train_eval": (X_train_full, Y_train_full),
        "valid_eval": None,
        "test": (X_test, Y_test),
        "has_valid_eval": False,
    }


def build_parameter_configs(param_grid, strategy):
    base_config = {k: v[0] for k, v in param_grid.items()}
    configs = []
    if strategy == "grid":
        keys = list(param_grid.keys())
        for values in itertools.product(*param_grid.values()):
            config = dict(zip(keys, values))
            config["_varying"] = "grid"
            configs.append(config)
        return configs, base_config
    # one-vs-rest: vary one parameter at a time while others stay at baseline
    for key, values in param_grid.items():
        for value in values:
            config = base_config.copy()
            config[key] = value
            config["_varying"] = key
            configs.append(config)
    return configs, base_config


def evaluate_configuration(
    qldgs_module, X, Y, opts, runs, test_size, valid_size, split_mode
):
    dim = X.shape[1]
    results = []
    for run in range(runs):
        split_data = create_data_splits(X, Y, run, split_mode, test_size, valid_size)
        X_train_fs, Y_train_fs = split_data["fs_train"]
        X_valid_fs, Y_valid_fs = split_data["fs_valid"]
        X_train_eval, Y_train_eval = split_data["train_eval"]
        valid_eval = split_data["valid_eval"]
        X_test, Y_test = split_data["test"]
        local_opts = opts.copy()
        local_opts["random_seed"] = run
        start_time = time.time()
        FS = qldgs_module.fs(
            X_train_fs.copy(), X_valid_fs.copy(), Y_train_fs.copy(), Y_valid_fs.copy(), local_opts
        )
        elapsed = time.time() - start_time
        selected = FS["sf"] == 1
        if not np.any(selected):
            selected = np.ones(dim, dtype=bool)
        run_metrics = {
            "nf": FS["nf"],
            "time": elapsed,
        }
        if split_data["has_valid_eval"]:
            X_valid_eval, Y_valid_eval = valid_eval
            valid_pred = classifier_method(
                X_train_fs[:, selected],
                X_valid_eval[:, selected],
                Y_train_fs,
                Y_valid_eval,
                local_opts,
            )
            valid_metrics = calculate_metrics(Y_valid_eval, valid_pred, "all")
            run_metrics.update(
                {
                    "acc_valid": valid_metrics["Accuracy"],
                    "f1_valid": valid_metrics["F1-Score (Macro)"],
                    "auc_valid": valid_metrics["ROC AUC (Macro)"],
                }
            )
        test_pred = classifier_method(
            X_train_eval[:, selected], X_test[:, selected], Y_train_eval, Y_test, local_opts
        )
        test_metrics = calculate_metrics(Y_test, test_pred, "all")
        run_metrics.update(
            {
                "acc_test": test_metrics["Accuracy"],
                "f1_test": test_metrics["F1-Score (Macro)"],
                "auc_test": test_metrics["ROC AUC (Macro)"],
            }
        )
        results.append(run_metrics)
    return results


def run_sensitivity(args):
    datasets = discover_datasets(args.datasets)
    if not datasets:
        raise RuntimeError("No datasets found under Dataset/")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be within (0,1)")
    if not 0 < args.valid_size < 1:
        raise ValueError("--valid-size must be within (0,1)")
    qldgs_module = load_qldgs_module()
    param_grid = {
        "phi": parse_numeric_list(args.phi, float),
        "ql_lr": parse_numeric_list(args.ql_lr, float),
        "ql_gamma": parse_numeric_list(args.ql_gamma, float),
        "interval_num": parse_numeric_list(args.interval_num, int),
        "interval_iterations": parse_numeric_list(args.interval_iterations, int),
        "T": parse_numeric_list(args.iterations, int),
    }
    configs, base_config = build_parameter_configs(param_grid, args.strategy)
    base_opts = {
        "maxLt": args.runs,
        "N": args.population,
        "T": base_config["T"],
        "phi": base_config["phi"],
        "interval_iterations": base_config["interval_iterations"],
        "cluster": ",",
        "classify": args.classifier,
        "split": args.split,
        "func": args.func,
        "knn_para": args.knn_k,
    }
    all_records = []
    for dataset in datasets:
        X, Y = load_dataset(dataset)
        for config_id, config in enumerate(configs):
            label = f"{config.get('_varying', 'grid')}->{config_id}"
            print(f"[{dataset}] Testing {label} with settings {config}")
            local_opts = base_opts.copy()
            local_opts.update({k: v for k, v in config.items() if not k.startswith("_")})
            results = evaluate_configuration(
                qldgs_module,
                X,
                Y,
                local_opts,
                args.runs,
                args.test_size,
                args.valid_size,
                args.split_mode,
            )
            for run_idx, metrics in enumerate(results):
                record = {
                    "dataset": dataset,
                    "run": run_idx,
                    "varying_param": config.get("_varying", "grid"),
                    "phi": local_opts["phi"],
                    "ql_lr": local_opts["ql_lr"],
                    "ql_gamma": local_opts["ql_gamma"],
                    "interval_num": local_opts["interval_num"],
                    "interval_iterations": local_opts["interval_iterations"],
                    "iterations": local_opts["T"],
                }
                record.update(metrics)
                all_records.append(record)
    if not all_records:
        raise RuntimeError("No experimental records were produced")
    results_df = pd.DataFrame(all_records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    group_cols = [
        "dataset",
        "varying_param",
        "phi",
        "ql_lr",
        "ql_gamma",
        "interval_num",
        "interval_iterations",
        "iterations",
    ]
    agg_kwargs = {
        "acc_test_mean": ("acc_test", "mean"),
        "acc_test_std": ("acc_test", "std"),
        "f1_test_mean": ("f1_test", "mean"),
        "f1_test_std": ("f1_test", "std"),
        "auc_test_mean": ("auc_test", "mean"),
        "auc_test_std": ("auc_test", "std"),
        "nf_mean": ("nf", "mean"),
        "nf_std": ("nf", "std"),
        "time_mean": ("time", "mean"),
    }
    if args.split_mode in ("train_val_test", "main_exvute"):
        agg_kwargs.update(
            {
                "acc_valid_mean": ("acc_valid", "mean"),
                "acc_valid_std": ("acc_valid", "std"),
                "f1_valid_mean": ("f1_valid", "mean"),
                "f1_valid_std": ("f1_valid", "std"),
                "auc_valid_mean": ("auc_valid", "mean"),
                "auc_valid_std": ("auc_valid", "std"),
            }
        )
    summary = results_df.groupby(group_cols).agg(**agg_kwargs).reset_index()
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved raw records to {output_path}")
    print(f"Saved summary statistics to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Parameter sensitivity sweeps for QLDGS-PSO-Elite"
    )
    parser.add_argument("--datasets", nargs="*", help="Dataset names without .csv suffix")
    parser.add_argument("--runs", type=int, default=5, help="Independent runs per setting")
    parser.add_argument("--population", type=int, default=20, help="Population size N")
    parser.add_argument(
        "--iterations", type=str, default="200,400,600", help="Comma separated T values"
    )
    parser.add_argument(
        "--interval-num",
        type=str,
        default="10,15,20",
        help="Comma separated interval counts L",
    )
    parser.add_argument(
        "--interval-iterations",
        type=str,
        default="20,30,40",
        help="Comma separated iteration counts per interval M",
    )
    parser.add_argument(
        "--ql-lr", type=str, default="0.1,0.2,0.4", help="Comma separated learning rates"
    )
    parser.add_argument(
        "--ql-gamma",
        type=str,
        default="0.0,0.3,0.6",
        help="Comma separated discount factors",
    )
    parser.add_argument(
        "--phi",
        type=str,
        default="0.95,0.98,0.99",
        help="Comma separated fitness-factor weights φ",
    )
    parser.add_argument("--classifier", type=str, default="knn", help="Classifier for evaluation")
    parser.add_argument("--knn-k", type=int, default=3, help="k used when classifier=knn")
    parser.add_argument("--split", type=int, default=0, help="Split strategy used in Fun")
    parser.add_argument("--func", type=int, default=0, help="Objective variant index")
    parser.add_argument(
        "--split-mode",
        choices=["train_val_test", "train_test", "main_exvute"],
        default="train_val_test",
        help="Dataset split type: 6:2:2 train/val/test, 8:2 train/test, or Main_exvute's shared 7:3 split",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples used as held-out test set (use 0.3 to mirror Main_exvute)",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.25,
        help="Validation fraction within the remaining data; used for both 6:2:2 and the internal "
        "split required by the 8:2 option",
    )
    parser.add_argument(
        "--strategy",
        choices=["grid", "one-vs-rest"],
        default="one-vs-rest",
        help="Sweep strategy for parameter grid",
    )
    parser.add_argument(
        "--output", type=str, default="Result/parameter_sensitivity.csv", help="CSV output path"
    )
    args = parser.parse_args()
    run_sensitivity(args)


if __name__ == "__main__":
    main()
