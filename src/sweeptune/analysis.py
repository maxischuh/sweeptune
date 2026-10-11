"""Frequency responses from impulse responses: windowing, smoothing, band levels, SNR."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from .deconv import IR_PRE_S

FREQS = np.geomspace(15.0, 20000.0, 500)  # ~48 points / octave


def fdw_response(ir: np.ndarray, fs: int, freqs: np.ndarray = FREQS, cycles: float | None = 15.0,
                 max_window_s: float = 0.5, min_window_s: float = 0.003,
                 pre_s: float = IR_PRE_S) -> np.ndarray:
    """Magnitude response in dB using a frequency-dependent window.

    For each frequency the IR (direct sound at `pre_s`) is windowed with a
    right half-Hann of `cycles` periods, so bass sees the room's steady state
    while treble is dominated by direct sound + early reflections. `cycles=None`
    uses the fixed `max_window_s` window everywhere (plain long-window response).
    """
    t0 = int(pre_s * fs)
    out = np.empty(len(freqs))
    n_max = len(ir)
    for i, f in enumerate(freqs):
        win_s = max_window_s if cycles is None else np.clip(cycles / f, min_window_s, max_window_s)
        L = min(int(win_s * fs), n_max - t0)
        w = np.ones(t0 + L)
        w[t0:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(L) / L)
        n = np.arange(t0 + L)
        h = np.dot(ir[: t0 + L] * w, np.exp(-2j * np.pi * f * n / fs))
        out[i] = 20 * np.log10(max(np.abs(h), 1e-12))
    return out


def points_per_octave(freqs: np.ndarray) -> float:
    return 1.0 / np.log2(freqs[1] / freqs[0])


def smooth(freqs: np.ndarray, db: np.ndarray, fraction: float) -> np.ndarray:
    """Power-domain fractional-octave smoothing (`fraction`=1/3 → third-octave)."""
    width = max(1, int(round(fraction * points_per_octave(freqs))))
    p = 10 ** (np.asarray(db) / 10)
    return 10 * np.log10(uniform_filter1d(p, size=width | 1, mode="nearest"))


def band_levels(freqs: np.ndarray, db: np.ndarray, centers, width_oct: float = 1.0) -> np.ndarray:
    """Mean power over ±width/2 octave around each centre, in dB."""
    lf = np.log2(freqs)
    p = 10 ** (np.asarray(db) / 10)
    out = []
    for c in centers:
        m = np.abs(lf - np.log2(c)) <= width_oct / 2
        out.append(10 * np.log10(p[m].mean()) if m.any() else np.nan)
    return np.array(out)


def power_average(curves) -> np.ndarray:
    arr = np.asarray(list(curves))
    return 10 * np.log10(np.mean(10 ** (arr / 10), axis=0))


def level_offset(freqs, db, ref_db, lo: float = 500.0, hi: float = 2000.0) -> float:
    """dB to add to `db` so its mean over [lo, hi] matches `ref_db`."""
    m = (freqs >= lo) & (freqs <= hi)
    return float(np.mean(np.asarray(ref_db)[m] - np.asarray(db)[m]))


def snr_by_band(rec: np.ndarray, fs: int, sweep_start: int, sweep_len: int, noise_end: int,
                centers) -> np.ndarray:
    """Per-octave SNR: power while the sweep plays vs. the silent lead-in."""
    noise = rec[int(0.1 * fs): max(int(0.1 * fs) + 1024, noise_end)]
    sig = rec[sweep_start: sweep_start + sweep_len]
    nper = 8192
    f, p_sig = signal.welch(sig, fs, nperseg=min(nper, len(sig)))
    _, p_noise = signal.welch(noise, fs, nperseg=min(nper, len(noise)))
    out = []
    for c in centers:
        m = (f >= c / np.sqrt(2)) & (f < c * np.sqrt(2))
        out.append(10 * np.log10(p_sig[m].sum() / max(p_noise[m].sum(), 1e-30)))
    return np.array(out)


def peak_dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(max(np.max(np.abs(x)), 1e-12)))


def is_clipped(x: np.ndarray) -> bool:
    return bool(np.max(np.abs(x)) >= 0.999)
