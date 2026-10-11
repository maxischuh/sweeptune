"""Target room curves (0 dB at 1 kHz) and adaptation to the speakers' bass extension."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import level_offset, points_per_octave, smooth
from .config import Config

TARGETS = ("harman-like", "bk-1974", "flat")


def _norm_1k(freqs, t):
    return t - np.interp(np.log10(1000.0), np.log10(freqs), t)


def harman_like(freqs, shelf_db=6.0, shelf_hz=105.0, tilt_db_per_oct=-1.0, tilt_start_hz=100.0):
    """Gentle downward tilt (~1 dB/oct above 100 Hz) plus a smooth bass low-shelf.

    Follows Toole/Olive: preferred in-room responses fall smoothly with frequency;
    the shelf adds the warmth found in Harman's listening tests.
    """
    f = np.asarray(freqs, dtype=float)
    tilt = tilt_db_per_oct * np.log2(np.maximum(f, tilt_start_hz) / 1000.0)
    shelf = shelf_db * 0.5 * (1 - np.tanh(1.5 * np.log2(f / shelf_hz)))
    return _norm_1k(f, tilt + shelf)


def bk_1974(freqs):
    """Brüel & Kjær (1974): flat to 200 Hz, -3 dB at 2 kHz, -6 dB at 20 kHz."""
    pts_f = np.log10([10, 200, 2000, 20000, 40000])
    pts_d = [0, 0, -3, -6, -9]
    return _norm_1k(freqs, np.interp(np.log10(freqs), pts_f, pts_d))


def flat(freqs):
    return np.zeros(len(freqs))


def from_csv(path: Path, freqs):
    data = np.loadtxt(path, delimiter=",", comments="#")
    return _norm_1k(freqs, np.interp(np.log10(freqs), np.log10(data[:, 0]), data[:, 1]))


def get_target(name: str, freqs, cfg: Config) -> np.ndarray:
    if name == "harman-like":
        return harman_like(freqs, cfg.shelf_db, cfg.shelf_hz, cfg.tilt_db_per_oct)
    if name == "bk-1974":
        return bk_1974(freqs)
    if name == "flat":
        return flat(freqs)
    path = Path(name)
    if path.suffix == ".csv" and path.exists():
        return from_csv(path, freqs)
    raise KeyError(f"unknown target {name!r}: use {', '.join(TARGETS)} or a freq,dB CSV path")


def rolloff_frequency(freqs, measured_db, target_db, deficit_db: float = 6.0,
                      start_hz: float = 200.0) -> float:
    """Speaker/room -6 dB bass extension relative to the target.

    Walking down from `start_hz` on an octave-smoothed, level-aligned curve,
    the first point where the deficit exceeds `deficit_db` and stays there for
    half an octave (so narrow room nulls don't count).
    """
    m = smooth(freqs, measured_db, 1.0)
    m = m + level_offset(freqs, m, target_db, 200.0, 2000.0)
    deficit = target_db - m
    half = int(0.5 * points_per_octave(freqs))
    idx = np.nonzero(freqs <= start_hz)[0]
    for i in idx[::-1]:
        if deficit[i] > deficit_db:
            lo = max(0, i - half)
            if np.all(deficit[lo: i + 1] > deficit_db):
                return float(freqs[min(i + 1, len(freqs) - 1)])
    return float(freqs[0])


def adapt_to_speaker(freqs, target_db, f_low: float) -> np.ndarray:
    """Let the target follow a 2nd-order roll-off below the speakers' extension."""
    if f_low <= freqs[0] * 1.01:
        return target_db
    return target_db - 10 * np.log10(1 + (f_low / np.asarray(freqs)) ** 4)


def tilt_advice(rt60_mid: float, cfg: Config) -> str | None:
    if not np.isfinite(rt60_mid):
        return None
    if rt60_mid < 0.25 and cfg.target in ("harman-like", "bk-1974"):
        return (f"Room is very dead (RT60 {rt60_mid:.2f} s above 500 Hz): direct sound dominates, "
                "so a steep tilt may sound dull. Try `tilt_db_per_oct = -0.5` or the `flat` target.")
    if rt60_mid > 0.7:
        return (f"Room is live (RT60 {rt60_mid:.2f} s): the measured treble is inflated by "
                "reverberation; treat the room before chasing the high bands with EQ.")
    return None
