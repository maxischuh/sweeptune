"""Graphic-EQ model (RBJ peaking filters) and the band-gain solver."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from .analysis import band_levels, level_offset, smooth
from .config import Config, band_label

MODEL_FS = 48000
RIDGE = 2.0  # cost per dB² of any band gain (keeps gains small, fixes level ambiguity)
BOOST_PENALTY = 4.0  # extra cost per dB² of boost: cuts preferred (boost costs 3x a cut)
PEAK_PENALTY = 20.0  # cost per point and dB² that a bass peak is pushed beyond target + PEAK_MARGIN_DB
PEAK_MARGIN_DB = 3.0
PEAK_MAX_HZ = 500.0


def peaking_db(freqs, fc: float, gain_db: float, q: float, fs: int = MODEL_FS) -> np.ndarray:
    """Magnitude (dB) of an RBJ cookbook peaking EQ."""
    if gain_db == 0:
        return np.zeros(len(freqs))
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b = np.array([1 + alpha * A, -2 * cw, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * cw, 1 - alpha / A])
    z = np.exp(-1j * 2 * np.pi * np.asarray(freqs) / fs)
    H = (b[0] + b[1] * z + b[2] * z**2) / (a[0] + a[1] * z + a[2] * z**2)
    return 20 * np.log10(np.abs(H))


def eq_curve(freqs, gains, bands, model: dict[float, dict]) -> np.ndarray:
    out = np.zeros(len(freqs))
    for c, g in zip(bands, gains):
        m = model[c]
        out += peaking_db(freqs, m["fc"], g * m.get("gain_scale", 1.0), m["q"])
    return out


def solve_weights(freqs, trust_mask=None) -> np.ndarray:
    """Room-dominated bass matters most; above 2 kHz only the broad tilt."""
    lf = np.log2(freqs)
    w = np.interp(lf, np.log2([500, 2000]), [1.0, 0.25])
    w[(freqs < 20) | (freqs > 20000)] = 0.0
    if trust_mask is not None:
        w[~trust_mask] = 0.0
    return w / max(w.mean(), 1e-9)


def eq_input_curve(freqs, db) -> np.ndarray:
    """Third-octave below 1 kHz, octave above 2 kHz (no chasing treble detail)."""
    a, b = smooth(freqs, db, 1 / 3), smooth(freqs, db, 1.0)
    x = np.clip(np.log2(np.asarray(freqs) / 1000.0), 0, 1)
    return (1 - x) * a + x * b


@dataclass
class BandLimits:
    lower: np.ndarray
    upper: np.ndarray
    reasons: dict[float, str] = field(default_factory=dict)


def band_limits(freqs, room_db, positions_db, cfg: Config, f_low: float, trust_low: float,
                trust_high: float, target_db=None) -> BandLimits:
    """Per-band bounds with the psychoacoustic guardrails."""
    bands = cfg.bands
    lower = np.full(len(bands), cfg.eq_min_db)
    upper = np.full(len(bands), min(cfg.max_boost_db, cfg.eq_max_db))
    reasons: dict[float, str] = {}
    fine = smooth(freqs, room_db, 1 / 6)
    broad = smooth(freqs, room_db, 1.0)
    spread = None
    if positions_db is not None and len(positions_db) >= 3:
        spread = np.std([smooth(freqs, p, 1 / 6) for p in positions_db], axis=0)
    lf = np.log2(freqs)
    excess = None
    if target_db is not None:
        excess = fine + level_offset(freqs, fine, target_db) - target_db
    for i, c in enumerate(bands):
        if c < trust_low or c > trust_high:
            lower[i] = upper[i] = 0.0
            reasons[c] = f"outside mic trust range ({trust_low:g}-{trust_high:g} Hz), left at 0"
            continue
        if c < f_low:
            upper[i] = 0.0
            reasons[c] = f"below speaker bass extension (~{f_low:.0f} Hz): no boost"
            continue
        region = np.abs(lf - np.log2(c)) <= 1 / 2
        dip = fine[region] - broad[region]
        if excess is not None and excess[region].max() > 3:
            f_pk = freqs[region][np.argmax(excess[region])]
            upper[i] = 0.0
            reasons[c] = f"resonance at {f_pk:.0f} Hz already {excess[region].max():.0f} dB above target: no boost"
        elif dip.min() < -6:
            f_null = freqs[region][np.argmin(dip)]
            upper[i] = 0.0
            reasons[c] = f"deep null near {f_null:.0f} Hz ({dip.min():.0f} dB): boosting won't fill it"
        elif spread is not None and spread[region].mean() > 4:
            upper[i] = 0.0
            reasons[c] = "position-dependent response (>4 dB spread): no boost"
    return BandLimits(lower, upper, reasons)


@dataclass
class EqResult:
    bands: tuple[float, ...]
    gains: np.ndarray  # rounded to the device step
    raw: np.ndarray
    offset: float
    predicted: np.ndarray  # room + eq + offset (level-aligned to the target)
    residual_bands: np.ndarray  # predicted - target per band (dB)
    headroom_db: float  # lower the S3 volume by this much
    reasons: dict[float, str]

    def copy_string(self) -> str:
        return format_eq(self.bands, self.gains)


def solve(freqs, room_db, target_db, cfg: Config, model: dict, limits: BandLimits,
          weights: np.ndarray) -> EqResult:
    """Fit the band gains to bring the room onto the target.

    The overall level is anchored at 500 Hz-2 kHz (where targets are defined)
    rather than solved for: a free level offset would let the solver "boost"
    a capped band by cutting all the others.
    """
    bands = cfg.bands
    room = eq_input_curve(freqs, room_db)
    offset = level_offset(freqs, room, target_db)
    need = target_db - room - offset
    sw = np.sqrt(weights)
    B = np.stack([peaking_db(freqs, model[c]["fc"], 6.0 * model[c].get("gain_scale", 1.0), model[c]["q"]) / 6.0
                  for c in bands], axis=1)
    n = len(bands)
    A = np.vstack([sw[:, None] * B, math.sqrt(RIDGE) * np.eye(n)])
    rhs = np.concatenate([sw * need, np.zeros(n)])
    fixed = limits.lower == limits.upper
    lo, hi = limits.lower.copy(), limits.upper.copy()
    hi[fixed] += 1e-6  # solvers need lo < hi
    x0 = lsq_linear(A, rhs, bounds=(lo, hi)).x

    # Fine (1/6-oct) view of the bass: a boost must not lift a resonance far
    # above the target even when the third-octave average says "too low".
    bass = (freqs <= PEAK_MAX_HZ) & (weights > 0)
    fine_excess = (smooth(freqs, room_db, 1 / 6) + offset - target_db)[bass]

    def residuals(g):
        eqc = eq_curve(freqs, g, bands, model)
        e = eqc - need
        over = np.maximum(fine_excess + eqc[bass] - PEAK_MARGIN_DB, 0)
        over *= g[np.argmin(np.abs(np.log2(np.asarray(bands)[:, None] / freqs[bass])), axis=0)] > 0
        return np.concatenate([sw * e, math.sqrt(RIDGE) * g, math.sqrt(BOOST_PENALTY) * np.maximum(g, 0),
                               math.sqrt(PEAK_PENALTY) * over])

    raw = least_squares(residuals, np.clip(x0, lo, hi), bounds=(lo, hi)).x
    g_raw = np.where(fixed, limits.lower, raw)
    step = cfg.eq_step_db
    gains = np.clip(np.round(g_raw / step) * step, limits.lower, limits.upper)
    gains = np.where(np.abs(gains) < 1e-9, 0.0, gains)
    eqc = eq_curve(freqs, gains, bands, model)
    predicted_full = room_db + eqc + offset
    resid = band_levels(freqs, smooth(freqs, predicted_full, 1 / 3) - target_db, bands)
    resid[band_levels(freqs, 10 * np.log10(weights + 1e-12), bands) < -60] = np.nan  # not measured
    in_band = (freqs >= 20) & (freqs <= 20000)
    peak = float(eqc[in_band].max())
    headroom = math.ceil(max(peak, 0.0) / step) * step
    return EqResult(bands, gains, g_raw, offset, predicted_full, resid, headroom, limits.reasons)


def fit_probe(freqs, diff_db, center: float, delta: float) -> dict:
    """Fit the device's real filter (fc, Q, gain scale) from a one-band change."""
    m = (freqs >= center / 4) & (freqs <= min(center * 4, 20000))
    f, d = freqs[m], diff_db[m]

    def res(p):
        g, q, o = p
        return peaking_db(f, center * 2**o, g, q) - d

    sol = least_squares(res, [delta, 1.41, 0.0], bounds=([-20, 0.3, -0.5], [20, 8.0, 0.5]))
    g, q, o = sol.x
    rms = float(np.sqrt(np.mean(sol.fun**2)))
    return {"fc": round(float(center * 2**o), 2), "q": round(float(q), 3),
            "gain_scale": round(float(g / delta), 3), "fit_rms_db": round(rms, 2)}


def format_eq(bands, gains) -> str:
    return " ".join(f"{band_label(c)}:{g:+.1f}" for c, g in zip(bands, gains))


def parse_eq(text: str | None, bands) -> np.ndarray:
    """Parse `31.5:-3 63:-4.5 1k:+1 …` or ten comma/space separated values."""
    gains = np.zeros(len(bands))
    if not text or text.strip().lower() in ("flat", "0", "none"):
        return gains
    if ":" in text:
        for item in re.split(r"[\s,;]+", text.strip()):
            if not item:
                continue
            k, v = item.split(":")
            k = k.lower().replace("hz", "")
            f = float(k[:-1]) * 1000 if k.endswith("k") else float(k)
            idx = int(np.argmin([abs(math.log2(f / b)) for b in bands]))
            if abs(math.log2(f / bands[idx])) > 0.1:
                raise ValueError(f"{k} is not one of the EQ bands")
            gains[idx] = float(v)
        return gains
    vals = [float(v) for v in re.split(r"[\s,;]+", text.strip()) if v]
    if len(vals) != len(bands):
        raise ValueError(f"expected {len(bands)} values, got {len(vals)}")
    return np.array(vals)
