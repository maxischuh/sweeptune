import numpy as np
import pytest
from scipy import signal

from sweeptune import acoustics
from sweeptune.analysis import FREQS, band_levels, fdw_response, power_average, smooth
from sweeptune.config import Config
from sweeptune.deconv import IR_PRE_S, extract_irs
from sweeptune.eq import peaking_db
from sweeptune.simulate import SimBackend, SimRoom, _peak_sos
from sweeptune.sweep import SweepPlan

FS = 48000


def short_plan(channels=("L",)):
    cfg = Config()
    cfg.gap_s = 2.5
    return SweepPlan.from_config(cfg, fs=FS, channels=channels, n=2**16)


def play(plan, irs, latency_s, level_dbfs=-12.0, noise_dbfs=-90.0, extra_s=5.0):
    sig = plan.signal(level_dbfs)
    out = np.zeros(len(sig) + int(extra_s * FS))
    lat = int(round(latency_s * FS))
    for ci, ch in enumerate(("L", "R")):
        if ch in irs and np.any(sig[:, ci]):
            y = signal.fftconvolve(sig[:, ci], irs[ch])
            n = min(len(y), len(out) - lat)
            out[lat: lat + n] += y[:n]
    out += 10 ** (noise_dbfs / 20) * np.random.default_rng(0).standard_normal(len(out))
    return out


def known_filter_ir(n=8192):
    imp = np.zeros(n)
    imp[0] = 1.0
    sos = np.vstack([_peak_sos(1000, 6.0, 2.0, FS), _peak_sos(80, -8.0, 3.0, FS),
                     signal.butter(2, 40, "highpass", fs=FS, output="sos")])
    return signal.sosfilt(sos, imp)


@pytest.mark.parametrize("latency_s", [0.0, 1.234, 4.9])
def test_deconvolution_recovers_filter_and_latency(latency_s):
    plan = short_plan()
    true_ir = known_filter_ir()
    rec = play(plan, {"L": true_ir}, latency_s)
    irs, info = extract_irs(rec, plan)
    assert abs(info["latency_s"] * FS - latency_s * FS) <= 1

    ref = np.concatenate([np.zeros(int(IR_PRE_S * FS)), true_ir, np.zeros(len(irs["L"]))])[: len(irs["L"])]
    got = smooth(FREQS, fdw_response(irs["L"], FS, cycles=None, max_window_s=0.15), 1 / 3) + 12.0  # sweep level
    want = smooth(FREQS, fdw_response(ref, FS, cycles=None, max_window_s=0.15), 1 / 3)
    m = (FREQS >= 30) & (FREQS <= 16000)
    assert np.max(np.abs(got[m] - want[m])) < 0.5

    # and the analysis itself reproduces the analytic filter shape (above the high-pass)
    analytic = peaking_db(FREQS, 1000, 6.0, 2.0) + peaking_db(FREQS, 80, -8.0, 3.0)
    m = (FREQS >= 200) & (FREQS <= 16000)
    assert np.max(np.abs(want[m] - smooth(FREQS, analytic, 1 / 3)[m])) < 0.5


def test_offline_recording_with_unknown_start_offset():
    plan = short_plan(("L", "R"))
    irs = {"L": known_filter_ir(), "R": 0.5 * known_filter_ir()}
    rec = play(plan, irs, latency_s=9.3, extra_s=12.0)  # recording started 9.3 s before playback
    out, info = extract_irs(rec, plan)
    assert abs(info["latency_s"] - 9.3) < 1e-3
    lvl_l = band_levels(FREQS, fdw_response(out["L"], FS), [1000])[0]
    lvl_r = band_levels(FREQS, fdw_response(out["R"], FS), [1000])[0]
    assert abs((lvl_l - lvl_r) - 6.02) < 0.3


def test_rt60_of_simulated_room_within_10_percent():
    room = SimRoom(rt60=0.5, noise_dbfs=-85)
    backend = SimBackend(room=room)
    plan = SweepPlan.from_config(Config(), fs=FS, channels=("L",))
    rec = backend.play_and_record(plan.signal(-12.0), FS)
    irs, _ = extract_irs(rec, plan)
    times = acoustics.decay_times(irs["L"], FS)
    for band in (500, 1000, 2000, 4000):
        assert acoustics.rt60(times, band) == pytest.approx(0.5, rel=0.10)


def test_early_reflections_found():
    h = SimRoom(noise_dbfs=-100).ir(1, "L")
    d = int(np.argmax(np.abs(h)))
    ir = np.concatenate([np.zeros(int(IR_PRE_S * FS) - d), h])  # direct sound at IR_PRE_S, as extract_irs does
    refl = acoustics.early_reflections(ir, FS)
    delays = [r["delay_ms"] for r in refl]
    for expected in (2.6, 5.1, 8.7):
        assert min(abs(d - expected) for d in delays) < 0.5


def test_smoothing_and_power_average():
    flat = np.zeros(len(FREQS))
    assert np.allclose(smooth(FREQS, flat, 1 / 3), 0)
    avg = power_average([flat, flat + 10 * np.log10(3)])
    assert np.allclose(avg, 10 * np.log10(2))


def test_live_balance_level_and_drift_corrected_distance():
    from sweeptune import balance as bal
    from sweeptune.measure import process_recording

    cfg = Config()
    room = SimRoom(r_gain_db=2.0)
    backend = SimBackend(room=room, clock_ppm=80.0)
    plan = SweepPlan.from_config(cfg, fs=FS, channels=("L", "R", "L"), n=2**17)
    plan.lead, plan.gap = FS, int(1.5 * FS)
    assert plan.labels == ("L", "R", "L2")
    irs, info = process_recording(backend.play_and_record(plan.signal(-18.0), FS), plan, cfg)
    b = bal.from_live(irs, info, FS)
    assert b.r_minus_l_db == pytest.approx(2.0, abs=0.3)
    assert b.distance_cm == pytest.approx(0.0004 * 343 * 100, abs=2.0)  # R is 0.4 ms farther in SimRoom
    raw_cm = info["arrival_ms"]["R"] / 1000 * 343 * 100
    assert abs(raw_cm - b.distance_cm) > 5  # 80 ppm drift would otherwise bias it by ~12 cm
    assert "toward L" in b.advice()
