"""Left/right channel balance: level and arrival-time difference at the seat.

Balance is judged on the band where voices and the stereo image live
(500 Hz-4 kHz) using the frequency-dependent-windowed response, so it is
dominated by the direct sound rather than by room modes. The mic's own
response cancels out because both channels are measured with the same mic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acoustics import OCTAVES, SPEED_OF_SOUND
from .analysis import FREQS, band_levels, fdw_response

BALANCE_BANDS = (500, 1000, 2000, 4000)
TOLERANCE_DB = 0.5


def channel_level(ir: np.ndarray, fs: int, cycles: float = 15.0, bands=BALANCE_BANDS) -> np.ndarray:
    return band_levels(FREQS, fdw_response(ir, fs, cycles=cycles), bands)


@dataclass
class Balance:
    r_minus_l_db: float  # positive: right is louder
    spread_db: float  # position-to-position spread (or L repeatability in live mode)
    per_octave: dict[int, float]  # R - L per octave band (63 Hz - 8 kHz)
    distance_cm: float | None = None  # positive: right speaker is farther away
    positions: int = 1

    def advice(self, step_db: float = 0.5) -> str:
        d = self.r_minus_l_db
        if abs(d) <= TOLERANCE_DB:
            return f"Balanced: R−L {d:+.1f} dB is within ±{TOLERANCE_DB} dB. Leave the balance where it is."
        louder, quieter = ("R", "L") if d > 0 else ("L", "R")
        amount = max(step_db, round(abs(d) / step_db) * step_db)
        text = (f"{louder} is {abs(d):.1f} dB louder → turn the amp's balance toward {quieter} by ~{amount:.1f} dB "
                f"(i.e. {louder} −{amount:.1f} dB).")
        if self.distance_cm is None:
            text += (" On a knob without dB markings, use `sweeptune balance --live` and turn until it reads "
                     f"within ±{TOLERANCE_DB} dB.")
        else:
            text += f" Turn a little, re-measure, and stop within ±{TOLERANCE_DB} dB."
        return text

    def distance_advice(self) -> str | None:
        if self.distance_cm is None:
            return None
        d = self.distance_cm
        if abs(d) < 3:
            return f"Both speakers are equidistant from the mic (difference {abs(d):.0f} cm)."
        far, near = ("R", "L") if d > 0 else ("L", "R")
        # With speakers at ±30°, a sideways seat move of x changes the L/R distance difference by ~x.
        return (f"The {far} speaker is {abs(d):.0f} cm farther away than {near}. For a centred image, move the "
                f"{far} speaker {abs(d):.0f} cm closer (or the seat ~{abs(d):.0f} cm sideways toward {far}) — "
                "fix this before trimming level with the balance control.")


def from_measurement(irs: dict[str, np.ndarray], fs: int) -> Balance | None:
    """Balance from a session measurement ('p01-L', 'p01-R', ... keys)."""
    positions = sorted({k.split("-")[0] for k in irs})
    diffs, octs = [], []
    for pos in positions:
        left, right = irs.get(f"{pos}-L"), irs.get(f"{pos}-R")
        if left is None or right is None:
            continue
        diffs.append(float(np.mean(channel_level(right, fs) - channel_level(left, fs))))
        octs.append(channel_level(right, fs, bands=OCTAVES) - channel_level(left, fs, bands=OCTAVES))
    if not diffs:
        return None
    per_oct = np.median(octs, axis=0)
    return Balance(float(np.median(diffs)), float(np.std(diffs)),
                   {b: round(float(v), 1) for b, v in zip(OCTAVES, per_oct)}, None, len(diffs))


def from_live(irs: dict[str, np.ndarray], info: dict, fs: int) -> Balance:
    """Balance from an L, R, L sequence at one position.

    Averaging the two L sweeps cancels slow level drift (AirPlay, AGC), and
    their spacing measures the player/recorder clock drift, which is removed
    from the R arrival time to get the true speaker distance difference.
    """
    l1, r, l2 = (channel_level(irs[k], fs) for k in ("L", "R", "L2"))
    l_mean = 10 * np.log10((10 ** (l1 / 10) + 10 ** (l2 / 10)) / 2)
    o_l = 10 * np.log10((10 ** (channel_level(irs["L"], fs, bands=OCTAVES) / 10)
                         + 10 ** (channel_level(irs["L2"], fs, bands=OCTAVES) / 10)) / 2)
    o_r = channel_level(irs["R"], fs, bands=OCTAVES)
    arr = info["arrival_ms"]
    dt_ms = arr["R"] - arr["L2"] / 2  # drift is linear in time: L2 is twice as far from L as R
    return Balance(float(np.mean(r - l_mean)), float(abs(np.mean(l1 - l2))),
                   {b: round(float(v), 1) for b, v in zip(OCTAVES, o_r - o_l)},
                   dt_ms / 1000 * SPEED_OF_SOUND * 100, 1)
