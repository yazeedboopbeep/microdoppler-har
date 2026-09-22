"""Hand-crafted micro-Doppler features for the classical baseline.

All features are computed from the normalised (0..1) 128 x 128 spectrogram
where rows are Doppler bins (-250 .. +250 Hz) and columns are time frames.
"""
from __future__ import annotations

import numpy as np

F_MAX = 250.0


def _axis(n: int) -> np.ndarray:
    return np.linspace(-F_MAX, F_MAX, n)


def spectrogram_features(s: np.ndarray) -> np.ndarray:
    f = _axis(s.shape[0])[:, None]
    p = s ** 2  # treat the normalised magnitude as a power-like weight
    p_t = p.sum(axis=0) + 1e-9
    centroid = (p * f).sum(axis=0) / p_t  # Hz per frame
    bandwidth = np.sqrt((p * (f - centroid) ** 2).sum(axis=0) / p_t)
    # envelope: highest / lowest Doppler with energy above a threshold per frame
    thr = 0.5
    above = s > thr
    upper = np.array([f[np.where(col)[0].max(), 0] if col.any() else 0.0 for col in above.T])
    lower = np.array([f[np.where(col)[0].min(), 0] if col.any() else 0.0 for col in above.T])
    pos = p[f[:, 0] > 0].sum()
    neg = p[f[:, 0] < 0].sum()
    energy_t = p.sum(axis=0)
    feats = [
        centroid.mean(), centroid.std(), centroid.max(), centroid.min(),
        bandwidth.mean(), bandwidth.std(), bandwidth.max(),
        upper.mean(), upper.max(), upper.std(),
        lower.mean(), lower.min(), lower.std(),
        (upper - lower).mean(), (upper - lower).max(),
        np.log(pos + 1e-9) - np.log(neg + 1e-9),
        energy_t.mean(), energy_t.std(), energy_t.max() / (energy_t.mean() + 1e-9),
        np.argmax(energy_t) / s.shape[1],  # when in the segment the peak happens
        s.mean(), s.std(),
        # coarse 8x8 pooled version of the spectrogram itself
        *s.reshape(8, s.shape[0] // 8, 8, s.shape[1] // 8).mean(axis=(1, 3)).ravel(),
    ]
    return np.asarray(feats, dtype=np.float32)


FEATURE_NAMES = (
    ["centroid_mean", "centroid_std", "centroid_max", "centroid_min",
     "bw_mean", "bw_std", "bw_max",
     "upper_mean", "upper_max", "upper_std",
     "lower_mean", "lower_min", "lower_std",
     "width_mean", "width_max", "pos_neg_log_ratio",
     "energy_mean", "energy_std", "energy_peak_ratio", "peak_time_frac",
     "spec_mean", "spec_std"]
    + [f"pool_{i}" for i in range(64)]
)
