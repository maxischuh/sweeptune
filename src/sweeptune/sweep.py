"""Exponential sine sweep (Farina) and the stereo measurement sequence.

One measurement is a single playback: lead silence (noise floor), then one sweep
per requested channel, separated by silent gaps long enough for the room decay
and the harmonic-distortion pre-echoes. Deconvolving the whole mono recording
with the mono sweep yields one impulse response peak per channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Config

CHANNELS = ("L", "R", "LR")


def ess(fs: int, n: int, f1: float, f2: float) -> np.ndarray:
    """Exponential sine sweep with short half-Hann fades, peak amplitude 1."""
    T = n / fs
    rate = T / np.log(f2 / f1)
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * f1 * rate * (np.exp(t / rate) - 1.0))
    fade_in = int(0.05 * fs)
    fade_out = int(0.01 * fs)
    x[:fade_in] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(fade_in) / fade_in)
    x[-fade_out:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(fade_out) / fade_out)
    return x


@dataclass
class SweepPlan:
    fs: int
    f1: float
    f2: float
    sweep: np.ndarray
    channels: tuple[str, ...]
    lead: int
    gap: int

    @classmethod
    def from_config(cls, cfg: Config, fs: int | None = None, channels=("L", "R"),
                    n: int | None = None) -> "SweepPlan":
        fs = fs or cfg.sample_rate
        f2 = min(cfg.f_end, 0.45 * fs)
        n = n or cfg.sweep_samples
        for ch in channels:
            if ch not in CHANNELS:
                raise ValueError(f"unknown channel {ch!r}, use one of {CHANNELS}")
        return cls(fs=fs, f1=cfg.f_start, f2=f2, sweep=ess(fs, n, cfg.f_start, f2),
                   channels=tuple(channels), lead=int(cfg.lead_s * fs), gap=int(cfg.gap_s * fs))

    @property
    def stride(self) -> int:
        return len(self.sweep) + self.gap

    @property
    def labels(self) -> tuple[str, ...]:
        """Unique name per sweep: a repeated channel gets a suffix (L, R, L2)."""
        seen: dict[str, int] = {}
        out = []
        for ch in self.channels:
            seen[ch] = seen.get(ch, 0) + 1
            out.append(ch if seen[ch] == 1 else f"{ch}{seen[ch]}")
        return tuple(out)

    def offsets(self) -> dict[str, int]:
        """Sample index at which each sweep starts in the signal, by label."""
        return {lab: self.lead + i * self.stride for i, lab in enumerate(self.labels)}

    def signal(self, level_dbfs: float) -> np.ndarray:
        """Stereo (n, 2) float32 playback signal."""
        amp = 10 ** (level_dbfs / 20)
        n = self.lead + len(self.channels) * self.stride
        out = np.zeros((n, 2), dtype=np.float32)
        for i, ch in enumerate(self.channels):
            start = self.lead + i * self.stride
            seg = slice(start, start + len(self.sweep))
            if ch in ("L", "LR"):
                out[seg, 0] = amp * self.sweep
            if ch in ("R", "LR"):
                out[seg, 1] = amp * self.sweep
        return out

    def duration_s(self) -> float:
        return (self.lead + len(self.channels) * self.stride) / self.fs


def export_wav(plan: SweepPlan, path: Path, level_dbfs: float = -12.0) -> Path:
    """Write the measurement sequence for offline recording (e.g. on the iPhone)."""
    sf.write(path, plan.signal(level_dbfs), plan.fs, subtype="PCM_24")
    return path
