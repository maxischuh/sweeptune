"""Measurement orchestration shared by the CLI commands (live and simulated)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import acoustics
from .analysis import FREQS, band_levels, fdw_response, is_clipped, peak_dbfs, power_average, snr_by_band
from .config import Config
from .deconv import extract_irs
from .eq import BandLimits, EqResult, band_limits, eq_curve, solve, solve_weights
from .mics import MicProfile
from .session import Measurement
from .sweep import SweepPlan
from .targets import adapt_to_speaker, get_target, rolloff_frequency

POSITIONS = [
    "listening seat, ear height (centre)",
    "30 cm to the left",
    "30 cm to the right",
    "30 cm forward",
    "30 cm back",
    "15 cm up",
    "15 cm down",
    "20 cm forward-left",
    "20 cm back-right",
]
SNR_BANDS = (63, 125, 250, 500, 1000, 2000, 4000, 8000)


class MeasurementError(RuntimeError):
    pass


def process_recording(rec: np.ndarray, plan: SweepPlan, cfg: Config) -> tuple[dict[str, np.ndarray], dict]:
    """Recording -> per-channel IRs + quality info (clipping, SNR per octave, latency)."""
    irs, info = extract_irs(rec, plan)
    fs = plan.fs
    first = plan.labels[0]
    start = info["peaks"][first]
    noise_end = start - int(0.05 * fs)
    snr = snr_by_band(rec, fs, start, len(plan.sweep), noise_end, SNR_BANDS)
    info.update({
        "clipped": is_clipped(rec),
        "input_peak_dbfs": peak_dbfs(rec),
        "snr_db": {str(b): round(float(s), 1) for b, s in zip(SNR_BANDS, snr)},
    })
    return irs, info


def quality_problems(info: dict, cfg: Config, trust_low: float, trust_high: float) -> list[str]:
    problems = []
    if info["clipped"]:
        problems.append(f"input clipped (peak {info['input_peak_dbfs']:.1f} dBFS): lower the level")
    low = [b for b, s in info["snr_db"].items()
           if trust_low <= float(b) <= trust_high and s < cfg.min_snr_db]
    if low:
        problems.append(f"SNR below {cfg.min_snr_db:.0f} dB in bands {', '.join(low)} Hz: raise the level or reduce noise")
    return problems


def short_plan(cfg: Config, fs: int) -> SweepPlan:
    plan = SweepPlan.from_config(cfg, fs=fs, channels=("L",), n=2**16)
    plan.lead, plan.gap = int(1.0 * fs), int(1.0 * fs)
    return plan


def level_probe(backend, cfg: Config, level: float) -> dict:
    """Short L sweep: input peak + broadband SNR, used for the level ramp."""
    plan = short_plan(cfg, backend.fs)
    rec = backend.play_and_record(plan.signal(level), plan.fs)
    irs, info = process_recording(rec, plan, cfg)
    snrs = [s for b, s in info["snr_db"].items() if 125 <= float(b) <= 4000]
    return {"peak_dbfs": info["input_peak_dbfs"], "clipped": info["clipped"],
            "snr_db": float(np.median(snrs)), "latency_s": info["latency_s"], "ir": irs["L"]}


def agc_check(backend, cfg: Config, level: float) -> tuple[bool, float]:
    """Two sweeps 6 dB apart must differ by 6 dB if no AGC/voice processing is active."""
    plan = short_plan(cfg, backend.fs)
    lv = []
    for lvl in (level, level - 6):
        rec = backend.play_and_record(plan.signal(lvl), plan.fs)
        irs, _ = process_recording(rec, plan, cfg)
        db = fdw_response(irs["L"], plan.fs, cycles=cfg.fdw_cycles)
        lv.append(np.mean(band_levels(FREQS, db, (250, 500, 1000, 2000, 4000))))
    diff = float(lv[0] - lv[1])
    return abs(diff - 6.0) <= 1.5, diff


def measure_positions(backend, cfg: Config, plan: SweepPlan, level: float, n_positions: int,
                      ask: Callable[[str], bool], on_position: Callable[[int], None] | None = None,
                      trust=(20.0, 20000.0), start_index: int = 1):
    """Run the guided multi-position measurement. Returns (irs, info, recordings)."""
    irs, infos, recs = {}, {}, {}
    p = start_index
    retries = 0
    while p < start_index + n_positions:
        label = POSITIONS[(p - 1) % len(POSITIONS)]
        if on_position:
            on_position(p)
        if retries == 0 and not ask(
                f"Position {p}/{start_index + n_positions - 1}: place the mic at the {label}. Ready?"):
            raise MeasurementError("aborted")
        rec = backend.play_and_record(plan.signal(level), plan.fs)
        try:
            ch_irs, info = process_recording(rec, plan, cfg)
        except RuntimeError as e:
            retries += 1
            if retries <= 3 and ask(f"{e}. Retry this position?"):
                continue
            raise MeasurementError(str(e)) from e
        problems = quality_problems(info, cfg, *trust)
        low_peak = [ch for ch, s in info["peak_snr_db"].items() if s < 30]
        if low_peak:
            problems.append(f"weak/unclear impulse peak on {', '.join(low_peak)} (AirPlay dropout?)")
        if problems:
            msg = "; ".join(problems)
            retries += 1
            if retries <= 3 and ask(f"Problem: {msg}. Retry this position?"):
                continue
            if info["clipped"]:
                raise MeasurementError(f"position {p}: {msg}")
            info["warnings"] = problems
        retries = 0
        tag = f"p{p:02d}"
        for ch, ir in ch_irs.items():
            irs[f"{tag}-{ch}"] = ir
        infos[tag] = info
        recs[tag] = rec
        p += 1
    return irs, infos, recs


# ---------------------------------------------------------------------- analysis
@dataclass
class RoomAnalysis:
    freqs: np.ndarray
    responses: dict[str, np.ndarray]  # mic-corrected FDW response per IR key
    average: np.ndarray  # power average of all keys (as measured, incl. applied EQ)
    room: np.ndarray  # average minus the applied EQ = the room/speakers alone
    per_channel: dict[str, np.ndarray]
    per_position: list[np.ndarray]  # room response per position (L/R power-averaged)


def analyze_measurement(m: Measurement, mic: MicProfile, cfg: Config, model: dict) -> RoomAnalysis:
    responses = {k: mic.correct(FREQS, fdw_response(ir, m.fs, cycles=cfg.fdw_cycles)) for k, ir in m.irs.items()}
    applied = eq_curve(FREQS, m.applied_eq, cfg.bands, model)
    avg = power_average(responses.values())
    per_channel = {ch: power_average([v for k, v in responses.items() if k.endswith(f"-{ch}")])
                   for ch in m.channels()}
    per_pos = [power_average([v for k, v in responses.items() if k.startswith(pos)]) - applied
               for pos in m.positions()]
    return RoomAnalysis(FREQS, responses, avg, avg - applied, per_channel, per_pos)


def pool(latest: RoomAnalysis, others: list[RoomAnalysis]) -> RoomAnalysis:
    """Combine the EQ-removed room responses of every iteration in a session.

    The setup doesn't change within a session, so all positions measure the
    same room: pooling them keeps the room estimate and the null/position
    guardrails stable when a verify pass uses fewer positions. `average`
    stays the latest measurement (what the listener hears now).
    """
    if not others:
        return latest
    per_pos = [p for ra in others for p in ra.per_position] + latest.per_position
    return RoomAnalysis(latest.freqs, latest.responses, latest.average, power_average(per_pos),
                        latest.per_channel, per_pos)


@dataclass
class Suggestion:
    target_name: str
    target: np.ndarray
    target_adapted: np.ndarray
    f_low: float
    limits: BandLimits
    result: EqResult
    current: np.ndarray


def low_snr_bands(m: Measurement, cfg: Config) -> list[float]:
    """EQ bands whose octave SNR fell below the minimum at any position."""
    worst: dict[float, float] = {}
    for info in m.info.values():
        for b, s in info.get("snr_db", {}).items():
            worst[float(b)] = min(worst.get(float(b), np.inf), s)
    return [c for c in cfg.bands if c in worst and worst[c] < cfg.min_snr_db]


def suggest(ra: RoomAnalysis, mic: MicProfile, cfg: Config, model: dict, current: np.ndarray,
            target_name: str | None = None, low_snr: list[float] | None = None) -> Suggestion:
    name = target_name or cfg.target
    target = get_target(name, ra.freqs, cfg)
    f_low = rolloff_frequency(ra.freqs, ra.room, target)
    adapted = adapt_to_speaker(ra.freqs, target, f_low)
    limits = band_limits(ra.freqs, ra.room, ra.per_position, cfg, f_low, mic.trust_low_hz, mic.trust_high_hz,
                         adapted)
    for c in low_snr or []:
        i = cfg.bands.index(c)
        limits.lower[i] = limits.upper[i] = 0.0
        limits.reasons[c] = f"SNR below {cfg.min_snr_db:.0f} dB: left at 0 (raise level / reduce noise)"
    weights = solve_weights(ra.freqs, mic.trust_mask(ra.freqs))
    result = solve(ra.freqs, ra.room, adapted, cfg, model, limits, weights)
    return Suggestion(name, target, adapted, f_low, limits, result, np.asarray(current, dtype=float))


def acoustics_summary(m: Measurement, ra: RoomAnalysis) -> dict:
    """Decay times averaged over all IRs, reflections of the first IR per channel, modes."""
    all_times = [acoustics.decay_times(ir, m.fs) for ir in m.irs.values()]
    bands = acoustics.OCTAVES
    rt = {}
    for b in bands:
        vals = [acoustics.rt60(t, b) for t in all_times]
        vals = [v for v in vals if np.isfinite(v)]
        rt[b] = float(np.median(vals)) if vals else float("nan")
    first = m.positions()[0]
    refl = {ch: acoustics.early_reflections(m.irs[f"{first}-{ch}"], m.fs) for ch in m.channels()}
    from .analysis import smooth

    md = acoustics.modes(ra.freqs, smooth(ra.freqs, ra.room, 1 / 12))
    mid = [rt[b] for b in (500, 1000, 2000) if np.isfinite(rt.get(b, np.nan))]
    return {"rt60": rt, "rt60_mid": float(np.mean(mid)) if mid else float("nan"),
            "reflections": refl, "modes": md}
