import numpy as np


def get_balanced_weights_per_class(y):
    weights = len(y) / (len(np.unique(y)) * np.bincount(y))
    return weights


def get_balanced_weights(y, c=None):
    if c is not None:
        y = np.array([y, c], dtype=str)
        y = np.array(list(map("_".join, zip(*y))))

    weights = np.zeros(len(y), dtype=np.float32)
    for idx, label in enumerate(y):
        weight = len(y) / (len(np.unique(y)) * np.count_nonzero(y == label))
        weights[idx] = weight
    return weights
