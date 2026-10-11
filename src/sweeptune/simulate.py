"""Synthetic listening room + streamer, for tests and `sweeptune --simulate`.

The simulated chain is: stereo signal -> S3 graphic EQ (true filter Q differs
from the default model so `eq probe` has something to learn) -> per-speaker
room IR (speaker roll-off, a 52 Hz mode, a position-dependent 115 Hz null,
early reflections, exponential reverb tail) -> mic coloration -> AirPlay-like
latency + noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .config import BANDS

TRUE_DEVICE_Q = 1.2


def _peak_sos(fc, gain_db, q, fs):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
    return np.hstack([b / a[0], a / a[0]])[None, :]


def eq_sos(gains, fs, q=TRUE_DEVICE_Q, bands=BANDS):
    secs = [_peak_sos(c, g, q, fs) for c, g in zip(bands, gains) if g != 0]
    return np.vstack(secs) if secs else None


MIC_COLORATION = {
    # (highpass Hz, [(fc, gain, q), ...])
    "audyssey": (None, []),
    "iphone17pro": (40, [(9000, 3.0, 1.0)]),
    "macbook-m1pro": (120, [(3000, 4.0, 1.5), (9000, -6.0, 0.7)]),
}


@dataclass
class SimRoom:
    fs: int = 48000
    rt60: float = 0.45
    latency_s: float = 1.8
    noise_dbfs: float = -80.0
    seed: int = 7
    speaker_hp_hz: float = 45.0
    mode_hz: float = 52.0
    mode_db: float = 9.0
    null_hz: float = 115.0
    r_gain_db: float = 1.0  # right speaker louder: something for `sweeptune balance` to find

    def ir(self, position: int, channel: str, length_s: float = 1.2) -> np.ndarray:
        fs = self.fs
        rng = np.random.default_rng(self.seed * 1000 + position * 10 + (channel == "R"))
        n = int(length_s * fs)
        h = np.zeros(n)
        d = int((0.0058 + (0.0004 if channel == "R" else 0.0)) * fs)
        h[d] = 1.0
        for t_ms, g_db in ((2.6, -6.0), (5.1, -9.0), (8.7, -11.0), (14.0, -14.0)):
            t = d + int((t_ms + rng.uniform(-0.3, 0.3)) / 1000 * fs)
            h[t] += 10 ** (g_db / 20) * rng.choice([-1, 1])
        start = d + int(0.012 * fs)
        t = np.arange(n - start) / fs
        tail = rng.standard_normal(n - start) * np.exp(-6.91 * t / self.rt60)
        h[start:] += 0.05 * tail
        sos = [signal.butter(2, self.speaker_hp_hz, "highpass", fs=fs, output="sos"),
               _peak_sos(self.mode_hz, self.mode_db, 6.0, fs),
               _peak_sos(self.null_hz, -6.0 - 2.5 * (position % 4), 5.0, fs),
               _peak_sos(3000, -1.5, 0.7, fs)]
        h = signal.sosfilt(np.vstack(sos), h)
        return 0.5 * h * (10 ** (self.r_gain_db / 20) if channel == "R" else 1.0)

    def mic_filter(self, mic: str, x: np.ndarray) -> np.ndarray:
        hp, peaks = MIC_COLORATION.get(mic, (None, []))
        secs = []
        if hp:
            secs.append(signal.butter(2, hp, "highpass", fs=self.fs, output="sos"))
        secs += [_peak_sos(fc, g, q, self.fs) for fc, g, q in peaks]
        return signal.sosfilt(np.vstack(secs), x) if secs else x


@dataclass
class SimBackend:
    """Drop-in for the live audio backend."""

    room: SimRoom = field(default_factory=SimRoom)
    mic: str = "audyssey"
    applied_eq: np.ndarray = field(default_factory=lambda: np.zeros(len(BANDS)))
    position: int = 1
    max_latency_s: float = 5.0
    clock_ppm: float = 30.0  # recorder clock vs AirPlay player clock
    output_name: str = "Simulated Fosi S3 (AirPlay)"
    input_name: str = "Simulated mic"

    @property
    def fs(self) -> int:
        return self.room.fs

    def play_and_record(self, sig: np.ndarray, fs: int) -> np.ndarray:
        assert fs == self.room.fs
        sos = eq_sos(self.applied_eq, fs)
        out = np.zeros(len(sig) + int(self.max_latency_s * fs))
        lat = int(self.room.latency_s * fs)
        for ci, ch in enumerate(("L", "R")):
            x = sig[:, ci].astype(float)
            if not np.any(x):
                continue
            if sos is not None:
                x = signal.sosfilt(sos, x)
            y = signal.fftconvolve(x, self.room.ir(self.position, ch))[: len(out) - lat]
            out[lat: lat + len(y)] += y
        out = self.room.mic_filter(self.mic, out)
        if self.clock_ppm:
            out = signal.resample(out, int(round(len(out) * (1 + self.clock_ppm * 1e-6))))
        rng = np.random.default_rng(self.position)
        out += 10 ** (self.room.noise_dbfs / 20) * rng.standard_normal(len(out))
        return out.astype(np.float32)

    def record(self, seconds: float, fs: int) -> np.ndarray:
        rng = np.random.default_rng(0)
        return (10 ** (self.room.noise_dbfs / 20) * rng.standard_normal(int(seconds * fs))).astype(np.float32)
