#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path
# Work around OpenMP runtime conflicts on Windows (libiomp5md.dll).
# Safe default: set only if not already provided by the environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)

# Make sure scripts/utils is importable when running from repo root
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils.data_handling import load_all_files, get_data_from_observation  # noqa: E402


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_channel_order_split():
    tasks = [
        'Relaxed1', 'Relaxed2', 'RelaxedTask1', 'RelaxedTask2',
        'StretchHold', 'LiftHold', 'HoldWeight', 'PointFinger',
        'DrinkGlas', 'CrossArms', 'TouchIndex', 'TouchNose',
        'Entrainment1', 'Entrainment2',
    ]
    channels_sorted = []
    for task in tasks:
        for wrist in ['LeftWrist', 'RightWrist']:
            for sensor in ['Time', 'Accelerometer', 'Gyroscope']:
                if sensor == 'Time':
                    channels_sorted.append('_'.join([task, wrist, sensor]))
                else:
                    for axis in ['X', 'Y', 'Z']:
                        channels_sorted.append('_'.join([task, wrist, sensor, axis]))
    return channels_sorted


def build_channel_order_raw(include_time: bool):
    tasks = [
        'Relaxed', 'RelaxedTask', 'StretchHold', 'LiftHold', 'HoldWeight',
        'PointFinger', 'DrinkGlas', 'CrossArms', 'TouchIndex', 'TouchNose',
        'Entrainment',
    ]
    wrists = ['LeftWrist', 'RightWrist']
    base_channels = [
        'Time',
        'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z',
        'Gyroscope_X', 'Gyroscope_Y', 'Gyroscope_Z',
    ]
    if not include_time:
        base_channels = [c for c in base_channels if c != 'Time']
    channels_sorted = []
    for task in tasks:
        for wrist in wrists:
            for ch in base_channels:
                channels_sorted.append('_'.join([task, wrist, ch]))
    return channels_sorted, tasks, wrists, base_channels


def _normalize_sensor_filter(sensor_filter: str | None):
    if sensor_filter is None:
        return None
    val = sensor_filter.lower()
    if val in ['both', 'all', 'none']:
        return None
    if val in ['acc', 'accel', 'acceleration', 'accelerometer']:
        return 'Accelerometer'
    if val in ['rot', 'gyro', 'rotation', 'gyroscope']:
        return 'Gyroscope'
    return None


def _apply_sensor_filter(data: np.ndarray, channels: np.ndarray, sensor_filter: str | None):
    sensor_key = _normalize_sensor_filter(sensor_filter)
    if sensor_key is None:
        return data, channels
    mask = pd.Series(channels).str.contains(sensor_key)
    return data[mask.values], channels[mask.values]


def _preprocessed_channel_names():
    # Preprocessed data always removed Time, LiftHold, PointFinger, TouchIndex
    base = np.array(build_channel_order_split())
    keep_mask = ~pd.Series(base).str.contains('Time|LiftHold|PointFinger|TouchIndex')
    return base[keep_mask]


def load_subject_matrix_split(df, movement_dir, remove_first=48, drop_regex='Time|LiftHold|PointFinger|TouchIndex',
                              sensor_filter: str | None = None):
    data, channels = get_data_from_observation(movement_dir, df)
    channels = list(channels)

    channels_sorted = build_channel_order_split()
    index_map = {ch: i for i, ch in enumerate(channels)}
    try:
        sorting_indices = [index_map[ch] for ch in channels_sorted]
    except KeyError as exc:
        raise ValueError(f'Missing channel in observation data: {exc}')

    data = data[sorting_indices]
    channels = np.array(channels)[sorting_indices]

    # Remove channels by regex (keeps current behavior by default)
    if drop_regex:
        keep_mask = ~pd.Series(channels).str.contains(drop_regex)
        data = data[keep_mask]
        channels = channels[keep_mask]

    # Remove first 0.48s (48 samples) to drop vibration notification
    if remove_first and remove_first > 0:
        data = data[:, remove_first:]

    data, channels = _apply_sensor_filter(data, channels, sensor_filter)

    return data.astype(np.float32)


def load_subject_matrix_raw_padded(df, movement_dir, pad_len=2048, include_time=True, remove_first=0,
                                   sensor_filter: str | None = None):
    channels_sorted, tasks, wrists, base_channels = build_channel_order_raw(include_time)
    index_map = {ch: i for i, ch in enumerate(channels_sorted)}
    data = np.zeros((len(channels_sorted), pad_len), dtype=np.float32)

    for _, meta_item in df.iterrows():
        task = meta_item['record_name']
        wrist = meta_item['device_location']
        if task not in tasks or wrist not in wrists:
            continue
        file_path = movement_dir + meta_item['file_name']
        record = np.loadtxt(file_path, dtype=np.float32, delimiter=",")
        if remove_first and remove_first > 0:
            record = record[remove_first:]
        record_len = min(record.shape[0], pad_len)

        for ch_idx, ch_name in enumerate(meta_item['channels']):
            if (not include_time) and ch_name == 'Time':
                continue
            key = '_'.join([task, wrist, ch_name])
            if key not in index_map:
                continue
            data[index_map[key], :record_len] = record[:record_len, ch_idx]

    data, channels_sorted = _apply_sensor_filter(data, np.array(channels_sorted), sensor_filter)
    return data


def load_subject_matrix_preprocessed(bin_path: Path, sensor_filter: str | None = None):
    data = np.fromfile(bin_path, dtype=np.float32)
    if data.size % 132 != 0:
        raise ValueError(f'Unexpected bin size for {bin_path}: {data.size} floats')
    data = data.reshape(132, data.size // 132)
    channels = _preprocessed_channel_names()
    data, _ = _apply_sensor_filter(data, channels, sensor_filter)
    return data.astype(np.float32)


def build_dataset(cache_path: Path, movement_dir: str, file_list: pd.DataFrame, force: bool, source: str,
                  preprocessed_dir: Path, raw_mode: str, include_time: bool, remove_first: int,
                  drop_regex: str, pad_len: int, sensor_filter: str | None):
    if cache_path.exists() and not force:
        cached = np.load(cache_path, allow_pickle=True)
        return cached['X'], cached['y'], cached['ids']

    X_list, y_list, id_list = [], [], []
    if source == 'raw':
        # Load all observation meta files once
        obs_dfs = load_all_files(movement_dir, dataframe=True)
        id_to_df = {}
        for df in obs_dfs:
            sid = str(df['subject_id'].iloc[0]).zfill(3)
            id_to_df[sid] = df

        for _, row in file_list.iterrows():
            sid = str(row['id']).zfill(3)
            if sid not in id_to_df:
                continue
            if raw_mode == 'pad':
                matrix = load_subject_matrix_raw_padded(
                    id_to_df[sid], movement_dir, pad_len=pad_len,
                    include_time=include_time, remove_first=remove_first,
                    sensor_filter=sensor_filter
                )
            else:
                matrix = load_subject_matrix_split(
                    id_to_df[sid], movement_dir,
                    remove_first=remove_first, drop_regex=drop_regex,
                    sensor_filter=sensor_filter
                )
            X_list.append(matrix)
            y_list.append(int(row['label']))
            id_list.append(sid)
    else:
        for _, row in file_list.iterrows():
            sid = str(row['id']).zfill(3)
            bin_path = preprocessed_dir / f'{sid}_ml.bin'
            if not bin_path.exists():
                continue
            matrix = load_subject_matrix_preprocessed(bin_path, sensor_filter=sensor_filter)
            X_list.append(matrix)
            y_list.append(int(row['label']))
            id_list.append(sid)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    ids = np.array(id_list, dtype=object)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, ids=ids)
    return X, y, ids


def prepare_file_list(file_list: pd.DataFrame, task: str):
    if task == 'binary':
        filtered = file_list[file_list['condition'].isin(['Healthy', "Parkinson's"])].copy()
        filtered['label'] = filtered['condition'].map({'Healthy': 0, "Parkinson's": 1})
        n_classes = 2
        target_names = ['Healthy', "Parkinson's"]
        return filtered, n_classes, target_names

    # 3-class default (Healthy vs Parkinson's vs Other)
    if 'label' not in file_list.columns:
        file_list = file_list.copy()
        file_list['label'] = file_list['condition'].map(
            lambda c: 0 if c == 'Healthy' else (1 if c == "Parkinson's" else 2)
        )
    n_classes = 3
    target_names = ['Healthy', "Parkinson's", 'Other']
    return file_list, n_classes, target_names


class MovementDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN1D(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, n_classes):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.append(pred.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    return (
        total_loss / total,
        correct / total,
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['3class', 'binary'], default='3class')
    parser.add_argument('--source', choices=['raw', 'preprocessed'], default='raw')
    parser.add_argument('--raw-mode', choices=['split', 'pad'], default='split')
    parser.add_argument('--raw-keep-all', action='store_true',
                        help='Use raw padded mode, keep all tasks/channels/time, and do not trim the start.')
    parser.add_argument('--include-time', action='store_true')
    parser.add_argument('--drop-tasks', type=str, default='')
    parser.add_argument('--remove-first', type=int, default=None)
    parser.add_argument('--pad-len', type=int, default=2048)
    parser.add_argument('--sensor-filter', choices=['both', 'acceleration', 'rotation'], default='both')
    parser.add_argument('--cache-path', type=str, default='')
    parser.add_argument('--force-rebuild', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Validation split ratio (per fold if cv-folds>1).')
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Train all epochs and use the final model (no early stopping).')
    args = parser.parse_args()

    set_seed(args.seed)

    base_dir = SCRIPT_DIR.parent
    movement_dir = str(base_dir / 'movement') + os.sep
    preprocessed_mov_dir = base_dir / 'preprocessed' / 'movement'
    file_list_path = base_dir / 'preprocessed' / 'file_list.csv'
    file_list = pd.read_csv(file_list_path)

    file_list, n_classes, target_names = prepare_file_list(file_list, args.task)

    if args.raw_keep_all:
        args.raw_mode = 'pad'

    if args.raw_mode == 'pad':
        include_time = True if not args.include_time else True
        drop_regex = ''
        remove_first = 0 if args.remove_first is None else args.remove_first
    else:
        include_time = True if args.include_time else False
        drop_regex = args.drop_tasks or 'Time|LiftHold|PointFinger|TouchIndex'
        remove_first = 48 if args.remove_first is None else args.remove_first

    sensor_filter = None if args.sensor_filter == 'both' else args.sensor_filter

    cache_path = args.cache_path
    if not cache_path:
        sensor_tag = args.sensor_filter
        if args.source == 'raw':
            cache_name = f'cnn_cache_raw_{args.raw_mode}_{args.task}_{sensor_tag}.npz'
        else:
            cache_name = f'cnn_cache_preprocessed_{args.task}_{sensor_tag}.npz'
        cache_path = str(Path('preprocessed') / cache_name)

    if args.source == 'raw':
        print(f'Raw mode: {args.raw_mode}, include_time={include_time}, '
              f'remove_first={remove_first}, pad_len={args.pad_len}, drop_regex={drop_regex}')

    X, y, ids = build_dataset(Path(cache_path), movement_dir, file_list, args.force_rebuild,
                              args.source, preprocessed_mov_dir, args.raw_mode, include_time,
                              remove_first, drop_regex, args.pad_len, sensor_filter)
    class_counts = np.bincount(y, minlength=n_classes)
    print(f'Dataset: X={X.shape}, y={y.shape}, class_counts={class_counts.tolist()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.cv_folds and args.cv_folds > 1:
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        metrics = {
            "accuracy": [],
            "balanced_accuracy": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "roc_auc": [],
        }
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
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

            # Standardize per-channel using training set only
            mean = X_train.mean(axis=(0, 2), keepdims=True)
            std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std
            X_test = (X_test - mean) / std

            train_ds = MovementDataset(X_train, y_train)
            val_ds = MovementDataset(X_val, y_val)
            test_ds = MovementDataset(X_test, y_test)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

            model = CNN1D(in_ch=X_train.shape[1], n_classes=n_classes).to(device)

            train_counts = np.bincount(y_train, minlength=n_classes)
            class_weights = (train_counts.sum() / (float(n_classes) * np.maximum(train_counts, 1))).astype(np.float32)
            criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            best_val = 0.0
            patience, patience_left = 7, 7
            best_state = None

            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device, n_classes)
                print(f'Fold {fold_idx:02d} Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.3f} | '
                      f'val loss {val_loss:.4f} acc {val_acc:.3f}')

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
                            print(f'Fold {fold_idx:02d} early stopping.')
                            break

            if (not args.no_early_stop) and (best_state is not None):
                model.load_state_dict(best_state)

            test_loss, test_acc, y_true, y_pred, y_prob = evaluate(
                model, test_loader, criterion, device, n_classes
            )
            metrics["accuracy"].append(accuracy_score(y_true, y_pred))
            metrics["balanced_accuracy"].append(balanced_accuracy_score(y_true, y_pred))
            metrics["f1"].append(f1_score(y_true, y_pred, average='macro'))
            metrics["precision"].append(precision_score(y_true, y_pred, average='macro'))
            metrics["recall"].append(recall_score(y_true, y_pred, average='macro'))
            try:
                if n_classes == 2:
                    metrics["roc_auc"].append(roc_auc_score(y_true, y_prob[:, 1]))
                else:
                    metrics["roc_auc"].append(roc_auc_score(y_true, y_prob, multi_class='ovr'))
            except Exception:
                pass

            print(f'Fold {fold_idx:02d} test loss {test_loss:.4f} acc {test_acc:.3f}')

        print('Cross-validation results (mean +/- std):')
        for key, vals in metrics.items():
            if len(vals) == 0:
                continue
            print(f'{key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')
        return

    # Fallback: single split (no CV)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    val_ratio = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, stratify=y_train, random_state=args.seed
    )

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    train_ds = MovementDataset(X_train, y_train)
    val_ds = MovementDataset(X_val, y_val)
    test_ds = MovementDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = CNN1D(in_ch=X_train.shape[1], n_classes=n_classes).to(device)

    train_counts = np.bincount(y_train, minlength=n_classes)
    class_weights = (train_counts.sum() / (float(n_classes) * np.maximum(train_counts, 1))).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 0.0
    patience, patience_left = 7, 7
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device, n_classes)
        print(f'Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.3f} | '
              f'val loss {val_loss:.4f} acc {val_acc:.3f}')

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
                    print('Early stopping.')
                    break

    if (not args.no_early_stop) and (best_state is not None):
        model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred, _ = evaluate(model, test_loader, criterion, device, n_classes)
    print(f'Test loss {test_loss:.4f} acc {test_acc:.3f}')
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))
    print('Classification report:')
    print(classification_report(y_true, y_pred, digits=3, target_names=target_names))


if __name__ == '__main__':
    main()
