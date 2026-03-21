import numpy as np
from transforms3d.axangles import axangle2mat
from scipy.interpolate import CubicSpline
import torch.utils.data as data


class DLDataset(data.Dataset):
    """
    A wrapper for the torch Dataset. Used to train deep learning models with the PADS dataset.
    Includes data augmentation when in 'train' mode.
    """
    def __init__(self, mode="train", data=None, labels=None):
        self.mode = mode
        self.data = data
        self.labels = labels
        self.length = data.shape[0]
        self.seq_len = data.shape[1]
        self.n_axes = 3

    def get_feature_matrix(self, index="all"):
        fv = []
        if index == "all":  # For all samples
            for sub_index in range(self.length):
                ts = self.load(sub_index)
                fv.append(ts.get_feature_matrix())
        elif isinstance(index, list):
            for sub_index in index:
                ts = self.load(sub_index)
                fv.append(ts.get_feature_matrix())
        else:
            ts = self.load(index)  # Only for certain sample
            fv = ts.get_feature_matrix()
        return fv

    def aug_rotate(self, x):
        idxs = []
        next_pos = 0
        for c in range(x.shape[0] // self.n_axes):
            idxs.append((next_pos, next_pos + self.n_axes))
            next_pos += self.n_axes

        x_new = []
        for idx in idxs:
            x_next = x[idx[0]:idx[1]]
            x_next = np.swapaxes(x_next, 0, 1)
            axis = np.random.uniform(low=-1, high=1, size=self.n_axes)
            angle = np.random.uniform(low=-np.pi, high=np.pi)
            mat = axangle2mat(axis, angle)
            x_next = np.matmul(x_next, mat)
            x_next = np.swapaxes(x_next, 0, 1)
            x_new.append(x_next)
        x_new = np.concatenate(x_new, axis=0)
        return x_new

    def aug_time_warp(self, x, sigma=0.2, knot=4):
        idxs = []
        next_pos = 0
        for c in range(x.shape[1] // self.seq_len):
            idxs.append((next_pos, next_pos + self.seq_len))
            next_pos += self.seq_len

        def generate_random_curve(X, sigma=0.2, knot=4):
            xx = (np.ones((1, 1)) * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))).transpose()
            yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
            x_range = np.arange(X.shape[0])
            cs = CubicSpline(xx[:, 0], yy[:, 0])
            return np.array([cs(x_range)]).transpose()

        def distort_timeseries(X, sigma):
            tt = generate_random_curve(X, sigma)  # Regard these samples around 1 as time intervals
            tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
            # Make the last value to have X.shape[0]
            t_scale = [(X.shape[0] - 1) / tt_cum[-1, 0]]
            tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
            return tt_cum

        x_new = []
        for idx in idxs:
            x_next = x[:, idx[0]:idx[1]]
            x_next = np.swapaxes(x_next, 0, 1)

            tt_new = distort_timeseries(x_next, sigma)

            x_range = np.arange(x_next.shape[0])

            x_next_new = np.apply_along_axis(lambda input: np.interp(x_range, tt_new[:, 0], input), 0, x_next)

            x_next_new = np.swapaxes(x_next_new, 0, 1)
            x_new.append(x_next_new)
        x_new = np.concatenate(x_new, axis=1)
        return x_new

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.data[idx]
        if self.mode == "train":
            data = self.aug_rotate(data)
            data = self.aug_time_warp(data)
        data = np.array(data, dtype="float32")
        label = np.int64(label)
        return data, label

    def __len__(self):
        return self.length
