from scipy.signal import welch
import numpy as np


def bandpower(x, srate=100, steps=100, cut=(1, 20), scale=True):
    """
    The power spectral density is computed in discrete 1 Hertz steps using Welch’s method.
    We only consider the values from 1 Hz to 19 Hz.
    """
    fs, pxx = welch(x, fs=srate, nperseg=steps, scaling="density")
    band_powers = pxx
    if cut is not None:
        band_powers = band_powers[cut[0]:cut[1]]
    if scale:
        band_powers = np.log10(band_powers)
    return band_powers


def std_windowed(x):
    """
    The time series is split into four segments of equal length.
    For each segment, the standard deviation is computed.
    """
    len_segm = int(len(x) / 4)
    stats = []
    for n in range(4):
        segm_data = x[n*len_segm:n*len_segm+len_segm]
        stats.extend([np.std(segm_data)])
    return stats


def abs_energy_windowed(x):
    """
    The time series is split into four segments of equal length.
    For each segment, the sum of absolute energy is computed.
    """
    len_segm = int(len(x) / 4)
    stats = []
    for n in range (4):
        segm_data = x[n*len_segm:n*len_segm+len_segm]
        stats.extend([np.sum(np.abs(segm_data))])
    return stats


def abs_max_windowed(x):
    """
    The time series is split into four segments of equal length.
    For each segment, the maximum absolute amplitude is computed.
    """
    len_segm = int(len(x) / 4)
    stats = []
    for n in range(4):
        segm_data = x[n*len_segm:n*len_segm+len_segm]
        stats.extend([np.max(np.abs(segm_data))])
    return stats


# The overall processing pipeline used on the PADS dataset.
processing_pipeline = [bandpower, std_windowed, abs_energy_windowed, abs_max_windowed]


def feature_extraction(x):
    """
    The complete feature extraction pipeline applied to the data matrix x, where each row encodes one sample.
    """
    features = []
    for func in processing_pipeline:
        features.append(np.apply_along_axis(func, 1, x))
    # Concatenate output to one feature vector per sample
    features = np.concatenate(features, axis=1)
    features = features.flatten()
    return features
