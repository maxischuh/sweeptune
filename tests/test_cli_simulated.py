"""End-to-end workflow against the synthetic room (no audio hardware)."""

import json
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from typer.testing import CliRunner

from sweeptune.cli import app
from sweeptune.simulate import SimBackend

runner = CliRunner()


def run(*args):
    result = runner.invoke(app, ["--simulate", *args])
    assert result.exit_code == 0, result.output
    return result.output


def latest_session() -> Path:
    return sorted(Path("sessions").iterdir())[-1]


def test_full_workflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("doctor")
    run("measure", "--positions", "5")
    out = run("eq", "suggest")
    assert "Copy string" in out
    s = latest_session()
    sugg = json.loads((s / "iter-00-baseline" / "suggestion.json").read_text())
    gains = np.array(sugg["gains"])
    assert gains.max() <= 4.0 and gains.min() >= -10.0
    assert sugg["gains"][-1] == 0  # 16 kHz outside the Audyssey trust range

    out = run("verify", "--positions", "3")
    assert "Converged" in out or "Next settings" in out
    v = json.loads((s / "iter-01-verify" / "measurement.json").read_text())
    assert v["applied_eq"] == sugg["gains"]

    out = run("eq", "probe", "125")
    model = json.loads(Path("streamer.json").read_text())["bands"]["125"]
    assert abs(model["q"] - 1.2) < 0.1  # the simulated S3's true Q

    out = run("plot", "--no-open")
    assert "frequency-response.png" in out
    run("plot", "--channels", "--no-open", "--out", "lr.png")
    assert Path("lr.png").stat().st_size > 10_000

    run("report")
    md = (s / "report.md").read_text()
    assert "Fosi S3" in md and "RT60" in md and "frequency-response.png" in md
    for png in ("frequency-response.png", "response.png", "rt60.png", "etc.png"):
        assert (s / png).stat().st_size > 10_000


def test_relative_mic_calibration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("mic", "relcal", "iphone17pro", "--ref", "audyssey", "--level", "-12")
    cal = Path("mics/iphone17pro.cal.txt").read_text().splitlines()
    rows = np.array([[float(x) for x in line.split()] for line in cal[1:]])
    at = lambda f: np.interp(np.log10(f), np.log10(rows[:, 0]), rows[:, 1])  # noqa: E731
    assert abs(at(1000)) < 0.2
    assert 1.5 < at(9000) < 4.0  # simulated iPhone presence peak (+3 dB @ 9 kHz)
    assert at(25) < -3  # simulated iPhone low cut
    assert json.loads(Path("mics/iphone17pro.json").read_text())["cal_file"] == "iphone17pro.cal.txt"


def test_offline_export_and_analyze(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("sweep", "export", "--out", "sweep.wav")
    sweep, fs = sf.read("sweep.wav", always_2d=True)
    rec = 0.25 * SimBackend(mic="iphone17pro").play_and_record(sweep.astype(np.float32), fs)  # mic gain
    rec = np.concatenate([np.zeros(int(3.7 * fs), dtype=np.float32), rec])  # recording started early
    g = gcd(fs, 44100)
    sf.write("phone.wav", resample_poly(rec, 44100 // g, fs // g), 44100)  # phone recorded at 44.1 kHz
    out = runner.invoke(app, ["analyze", "phone.wav", "--mic", "iphone17pro"])
    assert out.exit_code == 0, out.output
    m = json.loads((latest_session() / "iter-00-baseline" / "measurement.json").read_text())
    assert abs(m["info"]["p01"]["latency_s"] - (3.7 + 1.8)) < 0.01
    run("eq", "suggest")


def test_balance_commands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = run("balance", "--live")
    assert "R is 1.0 dB louder" in out and "farther away" in out
    run("measure", "--positions", "3")
    out = run("balance")
    assert "R − L = +1.0 dB" in out
    run("report")
    assert "Channel balance" in (latest_session() / "report.md").read_text()
