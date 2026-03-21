#!/usr/bin/env python
"""
PD vs HC classification using MOMENT foundation model.

This script loads a pretrained MOMENT checkpoint and fine-tunes it for binary
classification on the smartwatch movement dataset.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from itertools import product

# Work around OpenMP runtime conflicts on Windows (libiomp5md.dll).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    import optuna
except Exception as exc:  # pragma: no cover - optional dependency
    optuna = None
    OPTUNA_IMPORT_ERROR = exc
else:
    OPTUNA_IMPORT_ERROR = None

# Ensure scripts/ is importable.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_cnn_3class as base  # noqa: E402

try:
    from momentfm import MOMENTPipeline
except Exception as exc:  # pragma: no cover - runtime dependency guard
    MOMENTPipeline = None
    MOMENT_IMPORT_ERROR = exc
else:
    MOMENT_IMPORT_ERROR = None


def _build_model(
    model_id: str,
    n_channels: int,
    n_classes: int,
    reduction: str,
    enable_gradient_checkpointing: bool,
    device: torch.device,
):
    if MOMENTPipeline is None:
        raise RuntimeError(
            "momentfm is not installed or failed to import. "
            "Install with: pip install momentfm"
        ) from MOMENT_IMPORT_ERROR

    model = MOMENTPipeline.from_pretrained(
        model_id,
        model_kwargs={
            "task_name": "classification",
            "n_channels": n_channels,
            "num_class": n_classes,
            "reduction": reduction,
            "enable_gradient_checkpointing": enable_gradient_checkpointing,
        },
    )
    # The package requires explicit init() to switch from reconstruction to classification.
    model.init()
    return model.to(device)


def _set_train_mode(model: nn.Module, mode: str):
    if mode == "linear_probe":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    elif mode == "finetune":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unsupported train mode: {mode}")


def train_one_epoch(model, loader, optimizer, criterion, device, reduction: str):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        input_mask = torch.ones((X.size(0), X.size(-1)), dtype=torch.long, device=device)
        optimizer.zero_grad()
        outputs = model(x_enc=X, input_mask=input_mask, reduction=reduction)
        logits = outputs.logits
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device, reduction: str):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            input_mask = torch.ones((X.size(0), X.size(-1)), dtype=torch.long, device=device)
            outputs = model(x_enc=X, input_mask=input_mask, reduction=reduction)
            logits = outputs.logits
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.append(pred.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    return (
        total_loss / max(total, 1),
        correct / max(total, 1),
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def _compute_metrics(y_true, y_pred, y_prob):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "roc_auc": None,
    }
    try:
        if y_prob.shape[1] == 2:
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


def _require_optuna():
    if optuna is None:
        raise RuntimeError(
            "optuna is not installed or failed to import. "
            "Install with: pip install optuna"
        ) from OPTUNA_IMPORT_ERROR


def _run_hpo(args, X, y, n_classes, device: torch.device):
    _require_optuna()

    class_counts = np.bincount(y, minlength=n_classes)
    if args.cv_folds and args.cv_folds > 1:
        min_count = int(class_counts.min()) if len(class_counts) else 0
        if args.cv_folds > min_count:
            raise ValueError(
                f"cv-folds={args.cv_folds} is too high for smallest class count={min_count}. "
                "Lower cv-folds or increase samples."
            )
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        splits = list(skf.split(X, y))
    else:
        all_idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            all_idx,
            test_size=args.test_size,
            stratify=y,
            random_state=args.seed,
        )
        splits = [(train_idx, test_idx)]

    output_dir = Path(args.hpo_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.hpo_study_name,
        direction="maximize",
        storage=args.hpo_storage or None,
        load_if_exists=True,
    )

    def objective(trial):
        fold_scores = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train_full, y_train_full = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            X_train, X_val, y_train, y_val = _split_train_val(
                X_train_full, y_train_full, val_size=args.val_size,
                seed=args.seed, n_classes=n_classes
            )

            mean = X_train.mean(axis=(0, 2), keepdims=True)
            std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std
            X_test = (X_test - mean) / std

            batch_size = trial.suggest_categorical("batch_size", args.hpo_batch_sizes)
            train_ds = base.MovementDataset(X_train, y_train)
            val_ds = base.MovementDataset(X_val, y_val)
            test_ds = base.MovementDataset(X_test, y_test)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            reduction = trial.suggest_categorical("reduction", args.hpo_reductions)
            train_mode = trial.suggest_categorical("train_mode", args.hpo_train_modes)

            model = _build_model(
                model_id=args.foundation_model,
                n_channels=X_train.shape[1],
                n_classes=n_classes,
                reduction=reduction,
                enable_gradient_checkpointing=(not args.disable_grad_checkpointing),
                device=device,
            )
            _set_train_mode(model, train_mode)

            train_counts = np.bincount(y_train, minlength=n_classes)
            class_weights = (
                train_counts.sum() / (float(n_classes) * np.maximum(train_counts, 1))
            ).astype(np.float32)
            criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))

            lr = trial.suggest_float("lr", args.hpo_lr_min, args.hpo_lr_max, log=True)
            weight_decay = trial.suggest_float(
                "weight_decay", args.hpo_weight_decay_min, args.hpo_weight_decay_max, log=True
            )
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=lr,
                weight_decay=weight_decay,
            )

            best_val = 0.0
            patience_left = args.patience
            for epoch in range(1, args.epochs + 1):
                train_one_epoch(model, train_loader, optimizer, criterion, device, reduction)
                _, _, yv_true, yv_pred, _ = evaluate(model, val_loader, criterion, device, reduction)
                val_bal = balanced_accuracy_score(yv_true, yv_pred)

                trial.report(val_bal, step=((fold_idx - 1) * args.epochs + epoch))
                if trial.should_prune():
                    raise optuna.TrialPruned()

                improved = val_bal > best_val
                if improved:
                    best_val = val_bal
                    if not args.no_early_stop:
                        patience_left = args.patience
                elif not args.no_early_stop:
                    patience_left -= 1
                    if patience_left == 0:
                        break

            _, test_acc, y_true, y_pred, _ = evaluate(model, test_loader, criterion, device, reduction)
            test_bal = balanced_accuracy_score(y_true, y_pred)
            trial.set_user_attr(f"fold_{fold_idx}_test_bal", float(test_bal))
            trial.set_user_attr(f"fold_{fold_idx}_test_acc", float(test_acc))
            fold_scores.append(best_val)

            if device.type == "cuda":
                torch.cuda.empty_cache()

        return float(np.mean(fold_scores))

    study.optimize(
        objective,
        n_trials=args.hpo_trials,
        timeout=None if args.hpo_timeout <= 0 else args.hpo_timeout,
        gc_after_trial=True,
    )

    best = {"value": study.best_value, "params": study.best_params}
    best_path = output_dir / "best_params.json"
    with best_path.open("w") as f:
        json.dump(best, f, indent=2)
    trials_path = output_dir / "trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)

    print(f"Best balanced accuracy: {study.best_value:.4f}")
    print(f"Saved best params to {best_path}")
    print(f"Saved trials to {trials_path}")


def _run_grid_search(args, X, y, n_classes, device: torch.device):
    class_counts = np.bincount(y, minlength=n_classes)
    if args.cv_folds and args.cv_folds > 1:
        min_count = int(class_counts.min()) if len(class_counts) else 0
        if args.cv_folds > min_count:
            raise ValueError(
                f"cv-folds={args.cv_folds} is too high for smallest class count={min_count}. "
                "Lower cv-folds or increase samples."
            )
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        splits = list(skf.split(X, y))
    else:
        all_idx = np.arange(len(y))
        min_count = int(class_counts.min()) if len(class_counts) else 0
        if min_count < 2:
            train_idx, test_idx = all_idx, all_idx
        else:
            train_idx, test_idx = train_test_split(
                all_idx,
                test_size=args.test_size,
                stratify=y,
                random_state=args.seed,
            )
        splits = [(train_idx, test_idx)]

    grid_dir = Path(args.grid_output_dir)
    grid_dir.mkdir(parents=True, exist_ok=True)

    if "concat" in args.grid_reductions:
        print("Grid search: skipping reduction='concat' due to MOMENT head mismatch.")
    grid_reductions = ["mean"]
    configs = list(product(
        args.grid_lrs,
        args.grid_weight_decays,
        args.grid_batch_sizes,
        args.grid_train_modes,
        grid_reductions,
    ))
    if args.grid_max_trials and args.grid_max_trials > 0:
        configs = configs[: args.grid_max_trials]

    rows = []
    best_score = -float("inf")
    best_params = None

    for idx, (lr, weight_decay, batch_size, train_mode, reduction) in enumerate(configs, start=1):
        fold_scores = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            X_train_full, y_train_full = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            X_train, X_val, y_train, y_val = _split_train_val(
                X_train_full, y_train_full, val_size=args.val_size,
                seed=args.seed, n_classes=n_classes
            )

            mean = X_train.mean(axis=(0, 2), keepdims=True)
            std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std
            X_test = (X_test - mean) / std

            train_ds = base.MovementDataset(X_train, y_train)
            val_ds = base.MovementDataset(X_val, y_val)
            test_ds = base.MovementDataset(X_test, y_test)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            model = _build_model(
                model_id=args.foundation_model,
                n_channels=X_train.shape[1],
                n_classes=n_classes,
                reduction=reduction,
                enable_gradient_checkpointing=(not args.disable_grad_checkpointing),
                device=device,
            )
            _set_train_mode(model, train_mode)

            train_counts = np.bincount(y_train, minlength=n_classes)
            class_weights = (
                train_counts.sum() / (float(n_classes) * np.maximum(train_counts, 1))
            ).astype(np.float32)
            criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=lr,
                weight_decay=weight_decay,
            )

            best_val = 0.0
            patience_left = args.patience
            for epoch in range(1, args.epochs + 1):
                train_one_epoch(model, train_loader, optimizer, criterion, device, reduction)
                _, _, yv_true, yv_pred, _ = evaluate(
                    model, val_loader, criterion, device, reduction
                )
                val_bal = balanced_accuracy_score(yv_true, yv_pred)

                improved = val_bal > best_val
                if improved:
                    best_val = val_bal
                    if not args.no_early_stop:
                        patience_left = args.patience
                elif not args.no_early_stop:
                    patience_left -= 1
                    if patience_left == 0:
                        break

            _, test_acc, y_true, y_pred, _ = evaluate(
                model, test_loader, criterion, device, reduction
            )
            test_bal = balanced_accuracy_score(y_true, y_pred)
            fold_scores.append(best_val)

            rows.append({
                "trial": idx,
                "fold": fold_idx,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "train_mode": train_mode,
                "reduction": reduction,
                "val_balanced_accuracy": float(best_val),
                "test_balanced_accuracy": float(test_bal),
                "test_accuracy": float(test_acc),
            })

            if device.type == "cuda":
                torch.cuda.empty_cache()

        mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
        if mean_score > best_score:
            best_score = mean_score
            best_params = {
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "train_mode": train_mode,
                "reduction": reduction,
            }

    results_path = grid_dir / "grid_search_results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    best_path = grid_dir / "best_params.json"
    with best_path.open("w") as f:
        json.dump({"value": best_score, "params": best_params}, f, indent=2)

    print(f"Saved grid results to {results_path}")
    print(f"Saved best params to {best_path}")


def _split_train_val(X_train_full, y_train_full, val_size: float, seed: int, n_classes: int):
    if val_size <= 0:
        return X_train_full, X_train_full, y_train_full, y_train_full

    # If data is too small for stratified validation split, fall back to using
    # train as validation for smoke/debug runs.
    val_count = int(np.ceil(len(y_train_full) * val_size))
    class_counts = np.bincount(y_train_full, minlength=n_classes)
    if val_count < n_classes or np.any(class_counts < 2):
        return X_train_full, X_train_full, y_train_full, y_train_full

    try:
        return train_test_split(
            X_train_full, y_train_full, test_size=val_size,
            stratify=y_train_full, random_state=seed
        )
    except ValueError:
        return X_train_full, X_train_full, y_train_full, y_train_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["raw", "preprocessed"], default="preprocessed")
    parser.add_argument("--task", choices=["binary", "multiclass", "pd_dd"], default="binary",
                        help="binary=PD vs HC, pd_dd=PD vs other, multiclass=HC/PD/other.")
    parser.add_argument("--raw-mode", choices=["split", "pad"], default="split")
    parser.add_argument("--raw-keep-all", action="store_true")
    parser.add_argument("--include-time", action="store_true")
    parser.add_argument("--drop-tasks", type=str, default="")
    parser.add_argument("--remove-first", type=int, default=None)
    parser.add_argument("--pad-len", type=int, default=2048)
    parser.add_argument("--sensor-filter", choices=["both", "acceleration", "rotation"], default="both")
    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument("--force-rebuild", action="store_true")

    parser.add_argument("--foundation-model", type=str, default="AutonLab/MOMENT-1-small",
                        help="HuggingFace model id, e.g. AutonLab/MOMENT-1-small|base|large")
    parser.add_argument("--reduction", choices=["concat", "mean"], default="concat",
                        help="Channel reduction mode for MOMENT classification head.")
    parser.add_argument("--disable-grad-checkpointing", action="store_true",
                        help="Disable backbone gradient checkpointing. Usually faster on CPU.")
    parser.add_argument("--train-mode", choices=["linear_probe", "finetune"], default="linear_probe")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--hpo", action="store_true",
                        help="Run Optuna hyperparameter optimization instead of training.")
    parser.add_argument("--hpo-trials", type=int, default=20)
    parser.add_argument("--hpo-timeout", type=int, default=0,
                        help="Timeout in seconds (0 = no limit).")
    parser.add_argument("--hpo-study-name", type=str, default="moment_pd_hc")
    parser.add_argument("--hpo-storage", type=str, default="",
                        help="Optuna storage URL (optional).")
    parser.add_argument("--hpo-output-dir", type=str, default="./moment_training/optuna_pd_hc")
    parser.add_argument("--hpo-lr-min", type=float, default=5e-5)
    parser.add_argument("--hpo-lr-max", type=float, default=5e-4)
    parser.add_argument("--hpo-weight-decay-min", type=float, default=1e-6)
    parser.add_argument("--hpo-weight-decay-max", type=float, default=1e-3)
    parser.add_argument("--hpo-batch-sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--hpo-train-modes", type=str, nargs="+",
                        default=["linear_probe", "finetune"])
    parser.add_argument("--hpo-reductions", type=str, nargs="+", default=["concat", "mean"])
    parser.add_argument("--train-from-best", action="store_true",
                        help="Load best_params.json from --hpo-output-dir and run training with it.")
    parser.add_argument("--best-params-path", type=str, default="",
                        help="Explicit path to best_params.json (overrides --hpo-output-dir).")
    parser.add_argument("--grid-search", action="store_true",
                        help="Run a lightweight grid search instead of full training.")
    parser.add_argument("--grid-output-dir", type=str, default="./moment_training/grid_pd_hc")
    parser.add_argument("--grid-max-trials", type=int, default=0,
                        help="Cap number of grid configs to evaluate (0 = no cap).")
    parser.add_argument("--grid-lrs", type=float, nargs="+", default=[1e-4, 3e-4])
    parser.add_argument("--grid-weight-decays", type=float, nargs="+", default=[1e-6, 1e-4])
    parser.add_argument("--grid-batch-sizes", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--grid-train-modes", type=str, nargs="+",
                        default=["linear_probe", "finetune"])
    parser.add_argument("--grid-reductions", type=str, nargs="+", default=["concat", "mean"])
    parser.add_argument("--train-from-grid", action="store_true",
                        help="Load best_params.json from --grid-output-dir and run training with it.")
    parser.add_argument("--grid-best-params-path", type=str, default="",
                        help="Explicit path to grid best_params.json (overrides --grid-output-dir).")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--log-dir", type=str, default="./moment_training/runs_pd_hc")
    parser.add_argument("--output-dir", type=str, default="./moment_training/outputs_pd_hc")
    parser.add_argument("--max-samples-per-class", type=int, default=0,
                        help="If >0, subsample each class for quick debugging/smoke runs.")

    parser.add_argument("--out-csv", type=str, default="",
                        help="Output CSV path. Default: data/out/moment_pd_vs_hc_<source>_<sensor>.csv")
    args = parser.parse_args()

    base.set_seed(args.seed)

    base_dir = SCRIPT_DIR.parent
    dataset_root = base_dir / "1.0.0"
    if not dataset_root.exists():
        dataset_root = SCRIPT_DIR.parent

    movement_dir = str(dataset_root / "movement") + os.sep
    preprocessed_mov_dir = dataset_root / "preprocessed" / "movement"
    file_list = pd.read_csv(dataset_root / "preprocessed" / "file_list.csv")

    file_list, n_classes, _ = base.prepare_file_list(file_list, task=args.task)

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
        if args.task == "binary":
            task_tag = "binary_pd_vs_hc"
        elif args.task == "pd_dd":
            task_tag = "binary_pd_vs_other"
        else:
            task_tag = "multiclass_hc_pd_other"
        if args.source == "raw":
            cache_name = f"cnn_cache_raw_{args.raw_mode}_{task_tag}_{sensor_tag}.npz"
        else:
            cache_name = f"cnn_cache_preprocessed_{task_tag}_{sensor_tag}.npz"
        cache_path = str(Path("preprocessed") / cache_name)

    X, y, _ = base.build_dataset(
        Path(cache_path), movement_dir, file_list, args.force_rebuild, args.source,
        preprocessed_mov_dir, args.raw_mode, include_time, remove_first, drop_regex,
        args.pad_len, sensor_filter
    )

    if args.max_samples_per_class and args.max_samples_per_class > 0:
        selected_idx = []
        for label in range(n_classes):
            idx = np.where(y == label)[0]
            if len(idx) == 0:
                continue
            rng = np.random.default_rng(args.seed + label)
            take = min(args.max_samples_per_class, len(idx))
            selected_idx.extend(rng.choice(idx, size=take, replace=False).tolist())
        selected_idx = np.array(sorted(selected_idx), dtype=np.int64)
        X = X[selected_idx]
        y = y[selected_idx]
        print(f"Subsampled with max_samples_per_class={args.max_samples_per_class}")

    class_counts = np.bincount(y, minlength=n_classes)
    print(f"Dataset: X={X.shape}, y={y.shape}, class_counts={class_counts.tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.hpo:
        _run_hpo(args, X, y, n_classes, device)
        return
    if args.grid_search:
        _run_grid_search(args, X, y, n_classes, device)
        return

    if args.train_from_best:
        best_path = Path(args.best_params_path) if args.best_params_path else Path(args.hpo_output_dir) / "best_params.json"
        if not best_path.exists():
            raise FileNotFoundError(f"best_params.json not found at: {best_path}")
        with best_path.open("r") as f:
            best_data = json.load(f)
        best_params = best_data.get("params", {})
        if "lr" in best_params:
            args.lr = float(best_params["lr"])
        if "weight_decay" in best_params:
            args.weight_decay = float(best_params["weight_decay"])
        if "batch_size" in best_params:
            args.batch_size = int(best_params["batch_size"])
        if "train_mode" in best_params:
            args.train_mode = best_params["train_mode"]
        if "reduction" in best_params:
            args.reduction = best_params["reduction"]
        print(f"Loaded best params from {best_path}")

    if args.train_from_grid:
        grid_path = (
            Path(args.grid_best_params_path)
            if args.grid_best_params_path
            else Path(args.grid_output_dir) / "best_params.json"
        )
        if not grid_path.exists():
            raise FileNotFoundError(f"grid best_params.json not found at: {grid_path}")
        with grid_path.open("r") as f:
            grid_data = json.load(f)
        grid_params = grid_data.get("params", {})
        if "lr" in grid_params:
            args.lr = float(grid_params["lr"])
        if "weight_decay" in grid_params:
            args.weight_decay = float(grid_params["weight_decay"])
        if "batch_size" in grid_params:
            args.batch_size = int(grid_params["batch_size"])
        if "train_mode" in grid_params:
            args.train_mode = grid_params["train_mode"]
        if "reduction" in grid_params:
            args.reduction = grid_params["reduction"]
        print(f"Loaded grid params from {grid_path}")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "roc_auc": [],
    }
    fold_rows = []

    splits = []
    if args.cv_folds and args.cv_folds > 1:
        min_count = int(class_counts.min()) if len(class_counts) else 0
        if args.cv_folds > min_count:
            raise ValueError(
                f"cv-folds={args.cv_folds} is too high for smallest class count={min_count}. "
                "Lower cv-folds or increase samples."
            )
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        splits = list(skf.split(X, y))
    else:
        all_idx = np.arange(len(y))
        min_count = int(class_counts.min()) if len(class_counts) else 0
        if min_count < 2:
            train_idx, test_idx = all_idx, all_idx
        else:
            train_idx, test_idx = train_test_split(
                all_idx,
                test_size=args.test_size,
                stratify=y,
                random_state=args.seed,
            )
        splits = [(train_idx, test_idx)]

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train_full, y_train_full = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        X_train, X_val, y_train, y_val = _split_train_val(
            X_train_full, y_train_full, val_size=args.val_size, seed=args.seed, n_classes=n_classes
        )

        # Standardize per-channel with training split stats only.
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

        train_ds = base.MovementDataset(X_train, y_train)
        val_ds = base.MovementDataset(X_val, y_val)
        test_ds = base.MovementDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = _build_model(
            model_id=args.foundation_model,
            n_channels=X_train.shape[1],
            n_classes=n_classes,
            reduction=args.reduction,
            enable_gradient_checkpointing=(not args.disable_grad_checkpointing),
            device=device,
        )
        _set_train_mode(model, args.train_mode)

        train_counts = np.bincount(y_train, minlength=n_classes)
        class_weights = (train_counts.sum() / (float(n_classes) * np.maximum(train_counts, 1))).astype(np.float32)
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_val = 0.0
        patience_left = args.patience
        best_state = None

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, args.reduction
            )
            val_loss, val_acc, yv_true, yv_pred, _ = evaluate(
                model, val_loader, criterion, device, args.reduction
            )
            val_bal = balanced_accuracy_score(yv_true, yv_pred)
            global_step = (fold_idx - 1) * args.epochs + epoch
            writer.add_scalar("loss/train", train_loss, global_step)
            writer.add_scalar("loss/val", val_loss, global_step)
            writer.add_scalar("acc/train", train_acc, global_step)
            writer.add_scalar("acc/val", val_acc, global_step)
            writer.add_scalar("balanced_acc/val", val_bal, global_step)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
            print(
                f"Fold {fold_idx:02d} Epoch {epoch:02d} | "
                f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.3f} bal_acc {val_bal:.3f}"
            )

            improved = val_bal > best_val
            if improved:
                best_val = val_bal
                if not args.no_early_stop:
                    patience_left = args.patience
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save(
                        {"model_state": best_state, "val_bal": best_val, "epoch": epoch},
                        output_dir / f"best_fold_{fold_idx:02d}.pt",
                    )
            if not args.no_early_stop:
                if not improved:
                    patience_left -= 1
                    if patience_left == 0:
                        print(f"Fold {fold_idx:02d} early stopping.")
                        break

        if (not args.no_early_stop) and (best_state is not None):
            model.load_state_dict(best_state)

        test_loss, test_acc, y_true, y_pred, y_prob = evaluate(
            model, test_loader, criterion, device, args.reduction
        )
        fold_metrics = _compute_metrics(y_true, y_pred, y_prob)
        for key in metrics:
            metrics[key].append(fold_metrics[key])
        fold_rows.append({
            "row_type": "fold",
            "fold": fold_idx,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "test_loss": float(test_loss),
            "accuracy": float(fold_metrics["accuracy"]),
            "balanced_accuracy": float(fold_metrics["balanced_accuracy"]),
            "f1": float(fold_metrics["f1"]),
            "precision": float(fold_metrics["precision"]),
            "recall": float(fold_metrics["recall"]),
            "roc_auc": (None if fold_metrics["roc_auc"] is None else float(fold_metrics["roc_auc"])),
        })
        print(f"Fold {fold_idx:02d} test loss {test_loss:.4f} acc {test_acc:.3f}")

    print("Cross-validation results (mean +/- std):")
    summary = {}
    for key, vals in metrics.items():
        mean, std = _mean_std(vals)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
        if mean is None:
            print(f"{key}: NA")
        else:
            print(f"{key}: {mean:.4f} +/- {std:.4f}")

    out_csv = args.out_csv
    if not out_csv:
        source_tag = f"raw_{args.raw_mode}" if args.source == "raw" else "preprocessed"
        if args.task == "binary":
            task_tag = "pd_vs_hc"
        elif args.task == "pd_dd":
            task_tag = "pd_vs_other"
        else:
            task_tag = "hc_pd_other"
        out_csv = str(
            base_dir / "data" / "out" / f"moment_{task_tag}_{source_tag}_{args.sensor_filter}.csv"
        )
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_mean_row = {
        "row_type": "summary_mean",
        "fold": "mean",
        "n_train": None,
        "n_val": None,
        "n_test": None,
        "test_loss": None,
        "accuracy": summary["accuracy_mean"],
        "balanced_accuracy": summary["balanced_accuracy_mean"],
        "f1": summary["f1_mean"],
        "precision": summary["precision_mean"],
        "recall": summary["recall_mean"],
        "roc_auc": summary["roc_auc_mean"],
    }
    summary_std_row = {
        "row_type": "summary_std",
        "fold": "std",
        "n_train": None,
        "n_val": None,
        "n_test": None,
        "test_loss": None,
        "accuracy": summary["accuracy_std"],
        "balanced_accuracy": summary["balanced_accuracy_std"],
        "f1": summary["f1_std"],
        "precision": summary["precision_std"],
        "recall": summary["recall_std"],
        "roc_auc": summary["roc_auc_std"],
    }

    config_cols = {
        "foundation_model": args.foundation_model,
        "train_mode": args.train_mode,
        "reduction": args.reduction,
        "source": args.source,
        "raw_mode": args.raw_mode,
        "sensor_filter": args.sensor_filter,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "cv_folds": args.cv_folds,
    }

    rows = fold_rows + [summary_mean_row, summary_std_row]
    rows = [{**config_cols, **row} for row in rows]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")
    writer.close()


if __name__ == "__main__":
    main()
