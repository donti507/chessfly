"""Shared functions for the E-PG drift analysis (DANDI 000289, Basnak et al. 2025)."""
import numpy as np

def pb_angles(n_rois):
    """Basnak fitVonMises: full PB midline length = 2*pi*(15/8) of angle, wrapped to the circle.
    ROIs are treated as equally spaced along the midline."""
    d = np.arange(n_rois, dtype=float)
    return np.mod(d / d.max() * 2 * np.pi * (15 / 8), 2 * np.pi)

def bump_phase(dff):
    """Population-vector phase and strength per frame (frames x ROIs)."""
    th = pb_angles(dff.shape[1])
    x = np.clip(dff, 0, None)
    z = (x * np.exp(1j * th)).sum(1) / np.maximum(x.sum(1), 1e-9)
    return np.angle(z), np.abs(z)

def unwrap(a):
    return np.unwrap(a)

def msd_curve(segments, dt, max_lag_s):
    """Pooled mean squared displacement of unwrapped angle segments.
    Returns lags (s), msd, counts. Each segment contributes all its lags."""
    L = int(round(max_lag_s / dt))
    sums, counts = np.zeros(L + 1), np.zeros(L + 1)
    for s in segments:
        for k in range(1, min(L, len(s) - 1) + 1):
            d = s[k:] - s[:-k]
            sums[k] += (d ** 2).sum(); counts[k] += len(d)
    ok = counts > 0
    lags = np.arange(L + 1) * dt
    return lags[ok][1:], (sums[ok] / counts[ok])[1:], counts[ok][1:]

def fit_diffusion(lags, msd, max_fit_s):
    """MSD(t) = 2 D t + 2 sigma^2 fitted on lags <= max_fit_s. Returns D (rad^2/s), sigma (rad)."""
    m = lags <= max_fit_s
    A = np.c_[2 * lags[m], np.full(m.sum(), 2.0)]
    D, s2 = np.linalg.lstsq(A, msd[m], rcond=None)[0]
    return D, np.sqrt(max(s2, 0.0))
