"""Room acoustics from impulse responses: decay times, early reflections, modes."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .deconv import IR_PRE_S

OCTAVES = (63, 125, 250, 500, 1000, 2000, 4000, 8000)
SPEED_OF_SOUND = 343.0


def _octave_filter(ir: np.ndarray, fs: int, center: float) -> np.ndarray | None:
    hi = center * np.sqrt(2)
    if hi >= 0.45 * fs:
        return None
    sos = signal.butter(4, [center / np.sqrt(2), hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfilt(sos, ir)


def schroeder_db(h: np.ndarray, fs: int) -> np.ndarray:
    """Energy decay curve (dB re start) with noise-floor subtraction and truncation."""
    e = h**2
    start = int(np.argmax(uniform(e, int(0.005 * fs))[: int(0.08 * fs)]))
    e = e[start:]
    tail = e[-max(len(e) // 10, 1):]
    noise = float(tail.mean())
    env = uniform(e, int(0.01 * fs))
    above = np.nonzero(env <= 2 * noise)[0]
    cross = int(above[0]) if len(above) else len(e)
    cross = max(cross, int(0.05 * fs))
    ee = np.clip(e[:cross] - noise, 0, None)
    edc = np.cumsum(ee[::-1])[::-1]
    if edc[0] <= 0:
        return np.full(1, -np.inf)
    return 10 * np.log10(np.maximum(edc / edc[0], 1e-15))


def uniform(x: np.ndarray, n: int) -> np.ndarray:
    n = max(n, 1)
    return np.convolve(x, np.ones(n) / n, mode="same")


def _fit_decay(edc_db: np.ndarray, fs: int, top: float, bottom: float) -> float:
    idx = np.nonzero((edc_db <= top) & (edc_db >= bottom))[0]
    if len(idx) < 10 or edc_db.min() > bottom:
        return float("nan")
    t = idx / fs
    slope = np.polyfit(t, edc_db[idx], 1)[0]
    return float(-60.0 / slope) if slope < 0 else float("nan")


def decay_times(ir: np.ndarray, fs: int) -> dict[int, dict[str, float]]:
    """EDT, T20, T30 (seconds, extrapolated to 60 dB) per octave band."""
    out = {}
    for c in OCTAVES:
        h = _octave_filter(ir, fs, c)
        if h is None:
            continue
        edc = schroeder_db(h, fs)
        out[c] = {
            "edt": _fit_decay(edc, fs, 0.0, -10.0),
            "t20": _fit_decay(edc, fs, -5.0, -25.0),
            "t30": _fit_decay(edc, fs, -5.0, -35.0),
        }
    return out


def rt60(times: dict[int, dict[str, float]], band: int) -> float:
    """Best available decay estimate for a band: T30, else T20, else EDT."""
    t = times.get(band, {})
    for key in ("t30", "t20", "edt"):
        v = t.get(key, float("nan"))
        if np.isfinite(v):
            return v
    return float("nan")


def mean_rt60(times, bands=(500, 1000, 2000)) -> float:
    vals = [rt60(times, b) for b in bands]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def early_reflections(ir: np.ndarray, fs: int, window_ms: float = 20.0, threshold_db: float = -20.0,
                      max_count: int = 5) -> list[dict]:
    """Strongest reflections in the first `window_ms` of the energy-time curve."""
    sos = signal.butter(2, 300, btype="highpass", fs=fs, output="sos")
    h = signal.sosfilt(sos, ir)
    env = np.abs(signal.hilbert(h))
    t0 = int(IR_PRE_S * fs)
    lo, hi = max(0, t0 - int(0.001 * fs)), t0 + int(0.002 * fs)
    d = lo + int(np.argmax(env[lo:hi]))
    env_db = 20 * np.log10(np.maximum(env / env[d], 1e-9))
    start, stop = d + int(0.001 * fs), d + int(window_ms / 1000 * fs)
    seg = env_db[start:stop]
    peaks, props = signal.find_peaks(seg, height=threshold_db, prominence=3, distance=int(0.0005 * fs))
    order = np.argsort(props["peak_heights"])[::-1][:max_count]
    refl = []
    for i in sorted(peaks[order]):
        delay = (start + i - d) / fs
        refl.append({
            "delay_ms": round(delay * 1000, 2),
            "level_db": round(float(seg[i]), 1),
            "extra_path_cm": round(delay * SPEED_OF_SOUND * 100, 0),
        })
    return refl


def modes(freqs: np.ndarray, db: np.ndarray, f_max: float = 300.0) -> dict[str, list[dict]]:
    """Low-frequency peaks (resonances) and nulls on a (lightly smoothed) response.

    Heights/depths are relative to the octave-smoothed response, so the
    speakers' bass roll-off doesn't inflate them.
    """
    from scipy.ndimage import uniform_filter1d

    from .analysis import points_per_octave

    m = (freqs >= 20) & (freqs <= f_max)
    width = int(points_per_octave(freqs)) | 1
    rel = (db - uniform_filter1d(np.asarray(db, dtype=float), width, mode="nearest"))[m]
    f, y = freqs[m], db[m]
    out = {"peaks": [], "nulls": []}
    pk, _ = signal.find_peaks(rel, height=4, prominence=3)
    for i in pk:
        prom = rel[i]
        half = y[i] - 3
        left = i
        while left > 0 and y[left] > half:
            left -= 1
        right = i
        while right < len(y) - 1 and y[right] > half:
            right += 1
        bw = f[right] - f[left]
        out["peaks"].append({"freq": round(float(f[i]), 1), "prominence_db": round(float(prom), 1),
                             "q": round(float(f[i] / bw), 1) if bw > 0 else None})
    nl, _ = signal.find_peaks(-rel, height=6, prominence=3)
    for i in nl:
        out["nulls"].append({"freq": round(float(f[i]), 1), "depth_db": round(float(-rel[i]), 1)})
    return out
