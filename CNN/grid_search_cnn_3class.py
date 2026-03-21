#!/usr/bin/env python
"""
Grid search for CNN 3-class (HC/PD/DD) on movement time-series.
Uses train_cnn_3class.py utilities and runs stratified K-fold CV for each hyperparameter combo.
"""
import argparse
import itertools
import os
import sys
from pathlib import Path

# Work around OpenMP runtime conflicts on Windows (libiomp5md.dll).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

# Ensure scripts/ is importable
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_cnn_3class as base  # noqa: E402


def _parse_list(value: str, cast_fn):
    if not value:
        return []
    return [cast_fn(v.strip()) for v in value.split(",") if v.strip()]


def _compute_metrics(y_true, y_pred, y_prob, n_classes):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "roc_auc": None,
    }
    try:
        if n_classes == 2:
            out["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            out["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        out["roc_auc"] = None
    return out


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["3class", "binary"], default="3class")
    parser.add_argument("--binary-mode", choices=["pd_vs_hc", "pd_vs_dd"], default="pd_vs_hc",
                        help="Only used when --task binary.")
    parser.add_argument("--source", choices=["raw", "preprocessed"], default="preprocessed")
    parser.add_argument("--raw-mode", choices=["split", "pad"], default="split")
    parser.add_argument("--raw-keep-all", action="store_true")
    parser.add_argument("--include-time", action="store_true")
    parser.add_argument("--drop-tasks", type=str, default="")
    parser.add_argument("--remove-first", type=int, default=None)
    parser.add_argument("--pad-len", type=int, default=2048)
    parser.add_argument("--sensor-filter", choices=["both", "acceleration", "rotation"], default="both")
    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--no-early-stop", action="store_true")

    # Grid parameters (comma-separated)
    parser.add_argument("--lrs", type=str, default="0.001,0.0005")
    parser.add_argument("--batch-sizes", type=str, default="16,32")
    parser.add_argument("--epochs-list", type=str, default="30")
    parser.add_argument("--weight-decays", type=str, default="0.0")
    parser.add_argument("--select-metric", choices=[
        "accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"
    ], default="balanced_accuracy")
    parser.add_argument("--out-csv", type=str, default="",
                        help="Output CSV path. Default: data/out/grid_search_cnn_3class.csv")

    args = parser.parse_args()

    base.set_seed(args.seed)

    base_dir = SCRIPT_DIR.parent
    movement_dir = str(base_dir / "movement") + os.sep
    preprocessed_mov_dir = base_dir / "preprocessed" / "movement"
    file_list_path = base_dir / "preprocessed" / "file_list.csv"
    file_list = pd.read_csv(file_list_path)

    if args.task == "binary" and args.binary_mode == "pd_vs_dd":
        pd_mask = file_list["condition"] == "Parkinson's"
        dd_mask = ~file_list["condition"].isin(["Healthy", "Parkinson's"])
        filtered = file_list[pd_mask | dd_mask].copy()
        filtered["label"] = filtered["condition"].map(lambda c: 1 if c == "Parkinson's" else 0)
        file_list = filtered
        n_classes = 2
    else:
        # 3-class or HC vs PD (binary) default mapping from base
        file_list, n_classes, _ = base.prepare_file_list(file_list, args.task)

    if args.raw_keep_all:
        args.raw_mode = "pad"

    if args.raw_mode == "pad":
        include_time = True if not args.include_time else True
        drop_regex = ""
        remove_first = 0 if args.remove_first is None else args.remove_first
    else:
        include_time = True if args.include_time else False
        drop_regex = args.drop_tasks or "Time|LiftHold|PointFinger|TouchIndex"
        remove_first = 48 if args.remove_first is None else args.remove_first

    sensor_filter = None if args.sensor_filter == "both" else args.sensor_filter

    cache_path = args.cache_path
    if not cache_path:
        sensor_tag = args.sensor_filter
        task_tag = args.task
        if args.task == "binary":
            task_tag = f"{args.task}_{args.binary_mode}"
        if args.source == "raw":
            cache_name = f"cnn_cache_raw_{args.raw_mode}_{task_tag}_{sensor_tag}.npz"
        else:
            cache_name = f"cnn_cache_preprocessed_{task_tag}_{sensor_tag}.npz"
        cache_path = str(Path("preprocessed") / cache_name)

    X, y, _ = base.build_dataset(Path(cache_path), movement_dir, file_list, args.force_rebuild,
                                 args.source, preprocessed_mov_dir, args.raw_mode,
                                 include_time, remove_first, drop_regex, args.pad_len,
                                 sensor_filter)

    print(f"Dataset: X={X.shape}, y={y.shape}")

    lrs = _parse_list(args.lrs, float)
    batch_sizes = _parse_list(args.batch_sizes, int)
    epochs_list = _parse_list(args.epochs_list, int)
    weight_decays = _parse_list(args.weight_decays, float)
    if not lrs or not batch_sizes or not epochs_list or not weight_decays:
        raise ValueError("Grid lists cannot be empty.")

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    splits = list(skf.split(X, y))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    combos = list(itertools.product(lrs, batch_sizes, epochs_list, weight_decays))
    print(f"Grid size: {len(combos)} combos")

    for lr, batch_size, epochs, weight_decay in combos:
        fold_metrics = {k: [] for k in [
            "accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"
        ]}

        print(f"\nCombo: lr={lr}, batch={batch_size}, epochs={epochs}, weight_decay={weight_decay}")

        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train_full, y_train_full = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            if args.val_size > 0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_full, y_train_full, test_size=args.val_size,
                    stratify=y_train_full, random_state=args.seed
                )
            else:
                X_train, y_train = X_train_full, y_train_full
                X_val, y_val = X_train_full, y_train_full

            mean = X_train.mean(axis=(0, 2), keepdims=True)
            std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std
            X_test = (X_test - mean) / std

            train_ds = base.MovementDataset(X_train, y_train)
            val_ds = base.MovementDataset(X_val, y_val)
            test_ds = base.MovementDataset(X_test, y_test)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            model = base.CNN1D(in_ch=X_train.shape[1], n_classes=n_classes).to(device)

            train_counts = np.bincount(y_train, minlength=n_classes)
            class_weights = (train_counts.sum() / (float(n_classes) * np.maximum(train_counts, 1))).astype(np.float32)
            criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val = 0.0
            patience, patience_left = 7, 7
            best_state = None

            for epoch in range(1, epochs + 1):
                train_loss, train_acc = base.train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc, _, _, _ = base.evaluate(model, val_loader, criterion, device, n_classes)
                print(f"Fold {fold_idx:02d} Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.3f} | "
                      f"val loss {val_loss:.4f} acc {val_acc:.3f}")

                improved = val_acc > best_val
                if improved:
                    best_val = val_acc
                    if not args.no_early_stop:
                        patience_left = patience
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if not args.no_early_stop:
                    if not improved:
                        patience_left -= 1
                        if patience_left == 0:
                            print(f"Fold {fold_idx:02d} early stopping.")
                            break

            if (not args.no_early_stop) and (best_state is not None):
                model.load_state_dict(best_state)

            _, _, y_true, y_pred, y_prob = base.evaluate(
                model, test_loader, criterion, device, n_classes
            )
            m = _compute_metrics(y_true, y_pred, y_prob, n_classes)
            for key in fold_metrics:
                fold_metrics[key].append(m[key])

        row = {
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "weight_decay": weight_decay,
        }
        for key, vals in fold_metrics.items():
            mean, std = _mean_std(vals)
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        results.append(row)

    out_csv = args.out_csv
    if not out_csv:
        task_tag = args.task
        if args.task == "binary":
            task_tag = f"{args.task}_{args.binary_mode}"
        if args.source == "raw":
            source_tag = f"raw_{args.raw_mode}"
        else:
            source_tag = "preprocessed"
        sensor_tag = args.sensor_filter
        out_csv = str(
            base_dir
            / "data"
            / "out"
            / f"grid_search_cnn_{task_tag}_{source_tag}_{sensor_tag}.csv"
        )
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    # Print best by selected metric
    metric_mean_col = f"{args.select_metric}_mean"
    best_row = df.loc[df[metric_mean_col].idxmax()]
    print(f"\nBest by {args.select_metric}:\n{best_row}")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
