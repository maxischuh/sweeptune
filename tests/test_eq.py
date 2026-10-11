import numpy as np
import pytest

from sweeptune import eq
from sweeptune.analysis import FREQS
from sweeptune.config import BANDS, Config, load_streamer_model
from sweeptune.eq import BandLimits, band_limits, eq_curve, fit_probe, parse_eq, peaking_db, solve, solve_weights
from sweeptune.mics import parse_rew_cal, relative_cal, load_profile
from sweeptune.targets import adapt_to_speaker, bk_1974, harman_like, rolloff_frequency


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def model(cfg, tmp_path):
    return load_streamer_model(cfg, tmp_path / "none.json")


def wide_limits(lo=-10.0, hi=10.0):
    return BandLimits(np.full(len(BANDS), lo), np.full(len(BANDS), hi))


def test_solver_recovers_known_gains(cfg, model, monkeypatch):
    monkeypatch.setattr(eq, "BOOST_PENALTY", 0.0)
    monkeypatch.setattr(eq, "RIDGE", 0.05)
    known = np.array([0, -4, 3, -2, 0, 0, 0, 0, 0, 0], dtype=float)
    room = eq_curve(FREQS, known, BANDS, model)  # the room has the inverse of the wanted EQ
    target = np.zeros(len(FREQS))
    res = solve(FREQS, room, target, cfg, model, wide_limits(), solve_weights(FREQS))
    assert np.max(np.abs(res.gains + known)) <= 0.5


def test_gains_rounded_to_step_and_clamped(cfg, model):
    room = eq_curve(FREQS, [0, 0, 0, 9, 0, 0, 0, 0, 0, 0], BANDS, model)
    res = solve(FREQS, room, np.zeros(len(FREQS)), cfg, model, wide_limits(), solve_weights(FREQS))
    assert np.allclose(res.gains / 0.5, np.round(res.gains / 0.5))
    assert res.gains.min() >= -10 and res.gains.max() <= 10
    assert res.gains[3] < -5  # the broad 250 Hz bump is cut


def test_max_boost_cap_and_headroom(cfg, model):
    room = eq_curve(FREQS, [0, 0, 0, -9, 0, 0, 0, 0, 0, 0], BANDS, model)
    lim = band_limits(FREQS, room, None, cfg, f_low=20, trust_low=20, trust_high=20000)
    res = solve(FREQS, room, np.zeros(len(FREQS)), cfg, model, lim, solve_weights(FREQS))
    assert res.gains.max() <= cfg.max_boost_db
    assert res.gains[3] == cfg.max_boost_db
    assert res.headroom_db >= cfg.max_boost_db - 0.5


def test_no_boost_into_narrow_null(cfg):
    room = peaking_db(FREQS, 120, -18.0, 8.0)
    lim = band_limits(FREQS, room, None, cfg, f_low=20, trust_low=20, trust_high=20000)
    i = BANDS.index(125)
    assert lim.upper[i] == 0
    assert "null" in lim.reasons[125]


def test_no_boost_below_speaker_rolloff_and_outside_trust(cfg):
    room = -10 * np.log10(1 + (60 / FREQS) ** 4)  # 2nd-order roll-off at 60 Hz
    target = np.zeros(len(FREQS))
    f_low = rolloff_frequency(FREQS, room, target)
    assert 40 < f_low < 70
    lim = band_limits(FREQS, room, None, cfg, f_low, trust_low=20, trust_high=10000)
    assert lim.upper[BANDS.index(31.5)] == 0
    assert lim.lower[BANDS.index(16000)] == lim.upper[BANDS.index(16000)] == 0
    adapted = adapt_to_speaker(FREQS, target, f_low)
    assert adapted[np.argmin(abs(FREQS - 20))] < -10


def test_no_boost_into_resonance(cfg, model):
    target = harman_like(FREQS)
    room = target + peaking_db(FREQS, 55, 10.0, 6.0) + peaking_db(FREQS, 80, -6.0, 3.0)
    lim = band_limits(FREQS, room, None, cfg, 20, 20, 20000, target)
    res = solve(FREQS, room, target, cfg, model, lim, solve_weights(FREQS))
    assert res.gains[BANDS.index(63)] <= 0


def test_probe_fit_recovers_filter_q():
    diff = peaking_db(FREQS, 125, -6.0, 1.2)
    fit = fit_probe(FREQS, diff, 125, -6.0)
    assert fit["q"] == pytest.approx(1.2, abs=0.05)
    assert fit["gain_scale"] == pytest.approx(1.0, abs=0.05)


def test_targets_anchor_and_shape(cfg):
    h = harman_like(FREQS, shelf_db=6.0)
    i1k = np.argmin(abs(FREQS - 1000))
    assert abs(h[i1k]) < 0.05
    assert h[np.argmin(abs(FREQS - 31.5))] > 8  # shelf + tilt
    assert h[np.argmin(abs(FREQS - 16000))] < -3  # gentle treble tilt
    b = bk_1974(FREQS)
    assert abs(b[i1k]) < 0.05 and b[-1] < b[i1k]


def test_parse_eq_formats():
    g = parse_eq("31.5:-3 63:-4.5 1k:+1 16k:2", BANDS)
    assert list(g) == [-3, -4.5, 0, 0, 0, 1, 0, 0, 0, 2]
    assert list(parse_eq("1,2,3,4,5,6,7,8,9,10", BANDS)) == list(range(1, 11))
    assert not parse_eq("flat", BANDS).any()
    with pytest.raises(ValueError):
        parse_eq("700:1", BANDS)


def test_rew_cal_parsing_and_correction():
    text = '"Sens Factor =-1.2dB, SERNO: 7001234"\n* comment\n20.0\t-1.5\t0\n1000 0.0\n10000, 2.0\n'
    f, d = parse_rew_cal(text)
    assert list(f) == [20, 1000, 10000] and list(d) == [-1.5, 0, 2]
    mic = load_profile("audyssey")
    mic.calibration = [[20, -1.5], [1000, 0], [10000, 2.0]]
    corrected = mic.correct(FREQS, np.zeros(len(FREQS)))
    assert corrected[np.argmin(abs(FREQS - 10000))] == pytest.approx(-2.0, abs=0.05)


def test_relative_cal_normalized_at_1k():
    test = peaking_db(FREQS, 8000, 4.0, 1.0) + 3.0
    cal = relative_cal(FREQS, test, np.zeros(len(FREQS)))
    assert cal[np.argmin(abs(FREQS - 1000))] == pytest.approx(0, abs=0.1)
    assert cal[np.argmin(abs(FREQS - 8000))] == pytest.approx(3.9, abs=0.3)
