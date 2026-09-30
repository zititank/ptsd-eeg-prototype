"""Set A resting-state EEG features (frozen definitions of the v3 pipeline).

Set A is four spectral markers of the eyes-closed resting recording, computed on the occipital channels
(mean of O1 and O2 unless stated otherwise):

- `rest_IAF`            individual alpha frequency, Hz: the 7-14 Hz spectral peak, kept only if it is at
                        least 1.5x the median of the 7-14 Hz band outside +-1 Hz of the peak (else NaN).
- `rest_rel_alpha_O`    relative alpha power: 8-13 Hz power / 1-40 Hz power (dimensionless, 0-1).
- `rest_rel_alpha_IAF`  relative power in the individual alpha band IAF +- 2 Hz / 1-40 Hz power (NaN if no IAF).
- `rest_rel_beta_O`     relative beta power: 13-30 Hz power / 1-40 Hz power.

Pipeline per recording: robust centring, conditional 50 Hz notch, conditional 1-40 Hz band-pass, a shared
O1/O2 inter-quartile scale, 2 s epochs with 50% overlap, robust-z epoch rejection, Welch PSD per epoch,
average over epochs and occipital channels, band integrals (trapezoid rule).

All thresholds below were fixed before any cross-validation and never tuned on labels. Labels are never
used in this module.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch

from .io import CHANNELS, load_edf, rest_path

SET_A: list[str] = ["rest_IAF", "rest_rel_alpha_O", "rest_rel_alpha_IAF", "rest_rel_beta_O"]
OCCIPITAL: tuple[str, ...] = ("O1", "O2")

# Fixed engineering thresholds (locked before LOSO, never tuned on labels).
ROBUST_Z_MAX_ABS = 8.0            # epoch rejection: robust z of the epoch peak amplitude
ROBUST_Z_VARIANCE = 8.0           # epoch rejection: robust z of the epoch variance
NOTCH_HZ = 50.0                   # mains frequency
NOTCH_SPIKE_RATIO = 3.0           # notch only if the 50 Hz bin is >= 3x its neighbours
OUTSIDE_BAND_MAX_FRACTION = 0.05  # band-pass only if >= 5% of the power lies outside 1-40 Hz
APERIODIC_R2_MIN = 0.80           # aperiodic exponent kept only for a good log-log fit


def integrate(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Trapezoid integral of `y` over `x` (works on numpy 1.x and 2.x)."""
    y = np.asarray(y)
    x = np.asarray(x)
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z-score (0.6745 makes it comparable to a standard z for Gaussian data)."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-12:
        return np.zeros_like(x, dtype=float)
    return 0.6744897501960817 * (x - med) / mad


def bandpower(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """Absolute power in [low, high] Hz (inclusive), trapezoid rule; NaN if fewer than 2 bins."""
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    mask = (freqs >= low) & (freqs <= high)
    if np.count_nonzero(mask) < 2:
        return np.nan
    return float(integrate(psd[mask], freqs[mask]))


def relative_bandpower(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """Power in [low, high] Hz divided by 1-40 Hz power (dimensionless); NaN if undefined."""
    num = bandpower(freqs, psd, low, high)
    den = bandpower(freqs, psd, 1.0, 40.0)
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return np.nan
    return float(num / den)


def _safe_welch(data: np.ndarray, sfreq: float, seconds: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    n = data.shape[-1]
    nperseg = min(n, max(8, int(round(seconds * sfreq))))
    return welch(data, fs=sfreq, axis=-1, nperseg=nperseg)


def maybe_notch(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Apply a 50 Hz notch (Q = 30) only if a clear 50 Hz spectral spike exists."""
    nyquist = sfreq / 2.0
    if NOTCH_HZ >= nyquist - 1e-9:
        return data
    freqs, psd = _safe_welch(data, sfreq, seconds=4.0)
    idx = int(np.argmin(np.abs(freqs - NOTCH_HZ)))
    if idx < 2 or idx > len(freqs) - 3:
        return data
    neigh_idx = [i for i in range(max(0, idx - 3), min(len(freqs), idx + 4)) if i != idx]
    if not neigh_idx:
        return data
    local = np.nanmedian(psd[:, neigh_idx], axis=1)
    ratio = np.nanmedian(psd[:, idx] / np.maximum(local, 1e-30))
    if np.isfinite(ratio) and ratio >= NOTCH_SPIKE_RATIO:
        b, a = iirnotch(NOTCH_HZ, Q=30.0, fs=sfreq)
        return filtfilt(b, a, data, axis=-1)
    return data


def bandpass_1_40(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Zero-phase 4th-order Butterworth band-pass, 1-40 Hz (upper edge capped below Nyquist)."""
    nyquist = sfreq / 2.0
    high = min(40.0, nyquist * 0.98)
    if high <= 1.0:
        return data
    sos = butter(4, [1.0, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def preprocess_rest(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Robust centre, conditional 50 Hz notch, conditional 1-40 Hz band-pass. Input shape (6, n_samples)."""
    data = np.asarray(data, dtype=float).copy()
    if data.ndim != 2 or data.shape[0] != len(CHANNELS):
        raise ValueError(f"Expected EEG data shape (6, n_samples), got {data.shape}")
    if data.shape[1] < 8:
        raise ValueError("EEG recording is too short")
    data -= np.nanmedian(data, axis=1, keepdims=True)
    data = maybe_notch(data, sfreq)
    freqs, psd = _safe_welch(data, sfreq, seconds=4.0)
    total_mask = (freqs >= 0.5) & (freqs <= sfreq / 2.0)
    inside_mask = (freqs >= 1.0) & (freqs <= min(40.0, sfreq / 2.0))
    total = integrate(psd[:, total_mask], freqs[total_mask], axis=-1)
    inside = integrate(psd[:, inside_mask], freqs[inside_mask], axis=-1)
    outside_fraction = 1.0 - inside / np.maximum(total, 1e-30)
    if np.nanmedian(outside_fraction) >= OUTSIDE_BAND_MAX_FRACTION:
        data = bandpass_1_40(data, sfreq)
    return data


def make_rest_epochs(data: np.ndarray, sfreq: float) -> np.ndarray:
    """2 s epochs with 50% overlap; shape (n_epochs, 6, n_samples_per_epoch)."""
    epoch_len = int(round(2.0 * sfreq))
    if epoch_len < 2:
        raise ValueError(f"Invalid 2-second epoch length for fs={sfreq}")
    if data.shape[-1] < epoch_len:
        return np.empty((0, data.shape[0], epoch_len), dtype=float)
    step = max(1, epoch_len // 2)
    starts = range(0, data.shape[-1] - epoch_len + 1, step)
    return np.stack([data[:, s:s + epoch_len] for s in starts], axis=0)


def welch_epochs(epochs: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD of every epoch and channel (2 s segments, 50% overlap); returns (freqs, psd)."""
    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim != 3:
        raise ValueError(f"Expected epochs shape (n_epochs, 6, n_samples), got {epochs.shape}")
    if epochs.shape[0] == 0:
        return np.array([], dtype=float), np.empty((0, len(CHANNELS), 0), dtype=float)
    nperseg = min(epochs.shape[-1], max(8, int(round(2.0 * sfreq))))
    noverlap = min(nperseg // 2, nperseg - 1)
    return welch(epochs, fs=sfreq, axis=-1, nperseg=nperseg, noverlap=noverlap)


def mean_channel_psd(psd: np.ndarray, channel_names: Sequence[str]) -> np.ndarray:
    """Average PSD over epochs, then over the requested channels."""
    if psd.ndim != 3 or psd.shape[0] == 0:
        return np.array([], dtype=float)
    idx = [CHANNELS.index(name) for name in channel_names]
    per_channel = np.nanmean(psd[:, idx, :], axis=0)
    return np.nanmean(per_channel, axis=0)


def find_iaf(freqs: np.ndarray, spectrum: np.ndarray) -> float:
    """Individual alpha frequency (Hz): the 7-14 Hz peak if it is >= 1.5x the off-peak median, else NaN."""
    mask = (freqs >= 7.0) & (freqs <= 14.0)
    if np.count_nonzero(mask) < 3 or spectrum.size != freqs.size:
        return np.nan
    candidate_idx = np.where(mask)[0]
    vals = spectrum[mask]
    if not np.any(np.isfinite(vals)):
        return np.nan
    peak_idx = candidate_idx[int(np.nanargmax(vals))]
    peak_freq = float(freqs[peak_idx])
    baseline_mask = mask & ((freqs < peak_freq - 1.0) | (freqs > peak_freq + 1.0))
    baseline_vals = spectrum[baseline_mask]
    if baseline_vals.size == 0:
        return np.nan
    baseline = float(np.nanmedian(baseline_vals))
    peak = float(spectrum[peak_idx])
    if not np.isfinite(baseline) or baseline <= 0 or not np.isfinite(peak):
        return np.nan
    return peak_freq if peak >= 1.5 * baseline else np.nan


def _nan_spectral_features() -> dict[str, float]:
    return {"rest_IAF": np.nan, "rest_rel_alpha_O": np.nan, "rest_rel_alpha_IAF": np.nan,
            "rest_rel_beta_O": np.nan, "rest_AP_exponent": np.nan, "rest_AP_r2": np.nan, "rest_TBR_Fp": np.nan}


def rest_spectral_features(epochs: np.ndarray, sfreq: float,
                           occipital: Sequence[str] = OCCIPITAL) -> dict[str, float]:
    """Set A plus three descriptive companions (aperiodic exponent and fit R², frontal theta/beta ratio).

    Only Set A enters the model. `occipital` defaults to O1+O2; the O1-only variant is a predeclared
    sensitivity analysis.
    """
    if epochs.shape[0] == 0:
        return _nan_spectral_features()
    freqs, psd = welch_epochs(epochs, sfreq)
    if freqs.size == 0:
        return _nan_spectral_features()

    occ = mean_channel_psd(psd, list(occipital))
    frontal = mean_channel_psd(psd, ["Fp1", "Fp2"])
    iaf = find_iaf(freqs, occ)
    features = {
        "rest_IAF": iaf,
        "rest_rel_alpha_O": relative_bandpower(freqs, occ, 8.0, 13.0),
        "rest_rel_alpha_IAF": np.nan,
        "rest_rel_beta_O": relative_bandpower(freqs, occ, 13.0, 30.0),
        "rest_AP_exponent": np.nan,
        "rest_AP_r2": np.nan,
        "rest_TBR_Fp": np.nan,
    }
    if np.isfinite(iaf):
        features["rest_rel_alpha_IAF"] = relative_bandpower(freqs, occ, iaf - 2.0, iaf + 2.0)

    mask = (freqs >= 1.0) & (freqs <= 40.0) & np.isfinite(occ) & (occ > 0)
    if np.count_nonzero(mask) >= 5:
        x = np.log10(freqs[mask])
        y = np.log10(occ[mask])
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        features["rest_AP_r2"] = float(r2)
        if np.isfinite(r2) and r2 >= APERIODIC_R2_MIN:
            features["rest_AP_exponent"] = float(-slope)

    theta = relative_bandpower(freqs, frontal, 4.0, 8.0)
    beta = relative_bandpower(freqs, frontal, 13.0, 30.0)
    if np.isfinite(theta) and np.isfinite(beta) and beta > 0:
        features["rest_TBR_Fp"] = float(theta / beta)
    return features


def normalized_epochs(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess, scale all channels by the mean O1/O2 inter-quartile range, epoch, reject outlier epochs.

    Returns `(kept_epochs, keep_mask)`. The shared scale makes relative powers gain-invariant.
    """
    x = preprocess_rest(x, fs)
    common = np.mean([np.diff(np.percentile(x[c], [25, 75]))[0] for c in [0, 5]])
    if np.isfinite(common) and common > 1e-12:
        x = x / common
    epochs = make_rest_epochs(x, fs)
    if len(epochs):
        amp = np.nanmax(np.abs(epochs), axis=-1)
        var = np.nanvar(epochs, axis=-1)
        bad = (np.any(np.abs(robust_z(amp.ravel()).reshape(amp.shape)) > ROBUST_Z_MAX_ABS, axis=1)
               | np.any(np.abs(robust_z(var.ravel()).reshape(var.shape)) > ROBUST_Z_VARIANCE, axis=1)
               | np.any(~np.isfinite(amp), axis=1) | np.any(~np.isfinite(var), axis=1))
    else:
        bad = np.ones(0, dtype=bool)
    return epochs[~bad], ~bad


def extract_array(x: np.ndarray, fs: float,
                  occipital: Sequence[str] = OCCIPITAL) -> tuple[dict, np.ndarray, np.ndarray]:
    """Features of one (6, n_samples) recording. Returns `(features, keep_mask, kept_epochs)`.

    `features["scorable"]` is True when at least one epoch survives and both occipital relative powers
    are finite; unscorable subjects receive the prior fallback score downstream.
    """
    epochs, mask = normalized_epochs(x, fs)
    features = rest_spectral_features(epochs, fs, occipital=occipital)
    features.update(n_epochs=len(mask), n_kept=len(epochs),
                    rest_epoch_reject_frac=float(1 - mask.mean()) if len(mask) else 1.,
                    scorable=bool(len(epochs) and np.isfinite(features["rest_rel_alpha_O"])
                                  and np.isfinite(features["rest_rel_beta_O"])))
    return features, mask, epochs


def extract_subject(subject_dir: Path | str, occipital: Sequence[str] = OCCIPITAL) -> dict:
    """Set A (+ companions) for one subject folder. Raises OSError/ValueError/IndexError if unreadable."""
    x, fs = load_edf(rest_path(Path(subject_dir)))
    features, _, _ = extract_array(x, fs, occipital=occipital)
    return features
