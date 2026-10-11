"""Sweep deconvolution, peak/latency detection and channel split."""

from __future__ import annotations

import numpy as np
from scipy import fft

from .sweep import SweepPlan

IR_PRE_S = 0.01  # kept before the direct-sound peak
IR_LEN_S = 1.5  # kept after the peak (decay analysis)
PEAK_SEARCH_S = 0.05  # per-channel refinement window (speaker distance differences, clock drift)


def deconvolve(rec: np.ndarray, sweep: np.ndarray, fs: int, f1: float, f2: float) -> np.ndarray:
    """Regularized spectral division rec / sweep, band-limited to [f1, f2].

    The result is circular: the linear IR starts at index 0 (= where the sweep
    starts in the recording), harmonic distortion products wrap to the end.
    """
    n = fft.next_fast_len(len(rec) + len(sweep), real=True)
    X = fft.rfft(sweep, n)
    Y = fft.rfft(rec, n)
    f = fft.rfftfreq(n, 1 / fs)
    mag2 = np.abs(X) ** 2
    H = Y * np.conj(X) / (mag2 + 1e-10 * mag2.max())
    H *= _band_taper(f, f1, f2, fs)
    return fft.irfft(H, n)


def _band_taper(f: np.ndarray, f1: float, f2: float, fs: int) -> np.ndarray:
    w = np.ones_like(f)
    lo = f1 / 2
    w[f < lo] = 0.0
    m = (f >= lo) & (f < f1)
    w[m] = np.sin(0.5 * np.pi * np.log2(f[m] / lo)) ** 2
    hi = min(f2 * 1.25, fs / 2)
    m = f > f2
    if hi > f2:
        w[m] = np.cos(0.5 * np.pi * np.clip((f[m] - f2) / (hi - f2), 0, 1)) ** 2
    else:
        w[m] = 0.0
    return w


def locate_peaks(ir_full: np.ndarray, plan: SweepPlan, valid_len: int) -> dict[str, int]:
    """Find the direct-sound peak of every channel's IR in the deconvolved recording.

    The global maximum belongs to one of the channels; the hypothesis whose
    expected sibling peaks (spaced by the known sweep stride) carry the most
    energy wins. Works for live (latency up to a few s) and offline recordings
    (arbitrary start offset).
    """
    a = np.abs(ir_full[:valid_len])
    fs = plan.fs
    win = int(PEAK_SEARCH_S * fs)
    p = int(np.argmax(a))
    k = len(plan.channels)
    best, best_score = None, -1.0
    for i in range(k):
        s0 = p - i * plan.stride
        if s0 < 0 or s0 + (k - 1) * plan.stride >= valid_len:
            continue
        score = 0.0
        for j in range(k):
            c = s0 + j * plan.stride
            score += a[max(0, c - win): c + win].max()
        if score > best_score:
            best, best_score = s0, score
    if best is None:
        raise RuntimeError("could not locate the sweep in the recording (too short or no signal)")
    peaks = {}
    for j, ch in enumerate(plan.labels):
        c = best + j * plan.stride
        lo = max(0, c - win)
        peaks[ch] = lo + int(np.argmax(a[lo: c + win]))
    return peaks


def onset(ir: np.ndarray, peak: int, fs: int, rel_db: float = -12.0) -> int:
    """Earliest arrival within 3 ms before the peak above `rel_db` (direct sound
    can be weaker than an early reflection)."""
    a = np.abs(ir)
    lo = max(0, peak - int(0.003 * fs))
    thr = a[peak] * 10 ** (rel_db / 20)
    idx = np.nonzero(a[lo:peak + 1] >= thr)[0]
    return lo + int(idx[0]) if len(idx) else peak


def extract_irs(rec: np.ndarray, plan: SweepPlan) -> tuple[dict[str, np.ndarray], dict]:
    """Deconvolve a recording into one IR per channel.

    Returns ({channel: ir}, info) where each IR has its direct sound at
    IR_PRE_S seconds, and info holds latency, peak SNR and the arrival offset
    of each channel relative to the first (speaker distance difference plus
    clock drift; drift of a few tens of ppm does not affect the magnitude, and
    the per-channel peak search absorbs it, so no resampling is done).
    """
    fs = plan.fs
    ir_full = deconvolve(rec, plan.sweep, fs, plan.f1, plan.f2)
    peaks = locate_peaks(ir_full, plan, len(rec))
    pre, length = int(IR_PRE_S * fs), int(IR_LEN_S * fs)
    max_len = plan.stride - int(2.0 * fs)  # stay clear of the next sweep's harmonics
    length = min(length, max_len)
    irs, snr = {}, {}
    offsets = plan.offsets()
    for ch, p in peaks.items():
        s = onset(ir_full, p, fs)
        irs[ch] = ir_full[s - pre: s + length].copy() if s >= pre else np.pad(ir_full[: s + length], (pre - s, 0))
        noise = ir_full[s + length: s + length + int(0.5 * fs)]
        nrms = np.sqrt(np.mean(noise**2)) if len(noise) else 1e-12
        snr[ch] = float(20 * np.log10(np.abs(ir_full[p]) / max(nrms, 1e-12)))
    first = plan.labels[0]
    info = {
        "latency_s": float((peaks[first] - offsets[first]) / fs),
        "arrival_ms": {ch: round((peaks[ch] - peaks[first] - (offsets[ch] - offsets[first])) / fs * 1000, 2)
                       for ch in peaks},
        "peak_snr_db": snr,
        "peaks": {ch: int(v) for ch, v in peaks.items()},
    }
    return irs, info
