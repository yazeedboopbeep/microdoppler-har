"""Raw FMCW sweeps -> range-time map -> micro-Doppler spectrogram.

The steps follow the standard recipe used on this dataset (Fioranelli et al.):
  1. Hamming-windowed range FFT over each sweep.
  2. Moving-target indication: 4th-order Butterworth high-pass along slow time
     to strip static clutter (walls, furniture).
  3. Sum the range bins between 1.5 m and 10 m (bins 4..27 for a 400 MHz
     sweep, 1..7 for the 100 MHz files) and take an STFT along slow time with
     a 0.2 s Hamming window and 95 % overlap.
  4. Log-magnitude, clipped to a 40 dB dynamic range and normalised to [0, 1].
"""
from __future__ import annotations

import numpy as np
from scipy import signal

C = 299_792_458.0


def range_profiles(data: np.ndarray, n_fft: int | None = None) -> np.ndarray:
    """data: (fast-time, slow-time) complex. Returns (range bins, slow-time)."""
    nts = data.shape[0]
    n_fft = n_fft or nts
    win = np.hamming(nts)[:, None]
    rp = np.fft.fft(data * win, n=n_fft, axis=0)
    return rp[: n_fft // 2, :]  # positive ranges only


def mti(rp: np.ndarray, prf: float, cutoff_hz: float = 0.0075, order: int = 4) -> np.ndarray:
    """High-pass along slow time (axis 1). cutoff is expressed relative to Nyquist
    in the original MATLAB code; keep the same tiny value so results match the
    published preprocessing."""
    b, a = signal.butter(order, cutoff_hz, btype="high")
    return signal.lfilter(b, a, rp, axis=1)


def range_bins_for(bandwidth: float, r_min: float = 1.5, r_max: float = 10.0) -> tuple[int, int]:
    """Bin indices covering [r_min, r_max] metres. Most files are 400 MHz
    (0.375 m per bin) but 60 recordings in the West Cumbria campaign carry a
    100 MHz header (1.5 m per bin), and the target energy really does sit in
    bins 1..3 there, so the selection has to follow the header."""
    dr = C / (2 * bandwidth)
    return int(round(r_min / dr)), int(round(r_max / dr)) + 1


def micro_doppler(
    rp: np.ndarray,
    prf: float,
    bins: tuple[int, int] = (4, 27),
    window_s: float = 0.2,
    overlap: float = 0.95,
    n_fft: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (spectrogram dB, doppler axis Hz, time axis s)."""
    x = rp[bins[0] : bins[1], :].sum(axis=0)
    nperseg = int(round(window_s * prf))
    noverlap = int(round(overlap * nperseg))
    f, t, z = signal.stft(
        x,
        fs=prf,
        window="hamming",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=n_fft,
        return_onesided=False,
        boundary=None,
    )
    z = np.fft.fftshift(z, axes=0)
    f = np.fft.fftshift(f)
    s = 20 * np.log10(np.abs(z) + 1e-9)
    return s, f, t


def normalise_db(s: np.ndarray, dyn_range_db: float = 40.0) -> np.ndarray:
    s = s - s.max()
    s = np.clip(s, -dyn_range_db, 0.0)
    return (s + dyn_range_db) / dyn_range_db


def recording_to_spectrogram(rec, **kw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prf = 1.0 / rec.sweep_time
    rp = range_profiles(rec.data)
    rp = mti(rp, prf)
    kw.setdefault("bins", range_bins_for(rec.bandwidth))
    s, f, t = micro_doppler(rp, prf, **kw)
    return normalise_db(s), f, t


def range_time_map(rec) -> np.ndarray:
    prf = 1.0 / rec.sweep_time
    rp = mti(range_profiles(rec.data), prf)
    return 20 * np.log10(np.abs(rp) + 1e-9)


def range_axis(rec) -> np.ndarray:
    """Range (m) per FFT bin for a linear FMCW sweep."""
    n = rec.n_samples // 2
    dr = C / (2 * rec.bandwidth)
    return np.arange(n) * dr
