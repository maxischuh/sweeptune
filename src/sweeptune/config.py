"""Defaults, optional `sweeptune.toml` overrides and the learned streamer model."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

BANDS: tuple[float, ...] = (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)

CONFIG_FILE = Path("sweeptune.toml")
STREAMER_FILE = Path("streamer.json")


@dataclass
class Config:
    # Streamer EQ (Fosi S3)
    bands: tuple[float, ...] = BANDS
    eq_step_db: float = 0.5
    eq_min_db: float = -10.0
    eq_max_db: float = 10.0
    max_boost_db: float = 4.0
    default_q: float = 1.41

    # Audio routing
    output_device: str = "S3"  # name fragment; falls back to "AirPlay", then system default
    input_device: str = ""  # name fragment; empty = use the mic profile's patterns
    sample_rate: int = 48000

    # Sweep
    f_start: float = 20.0
    f_end: float = 20000.0
    sweep_samples: int = 2**18  # ~5.5 s at 48 kHz
    lead_s: float = 2.0  # leading silence (noise floor)
    gap_s: float = 3.0  # silence between L and R sweeps (decay + harmonics)
    max_latency_s: float = 5.0  # AirPlay latency headroom
    start_level_dbfs: float = -30.0
    max_level_dbfs: float = -3.0

    # Target
    target: str = "harman-like"
    shelf_db: float = 6.0
    shelf_hz: float = 105.0
    tilt_db_per_oct: float = -1.0

    # Analysis / acceptance
    fdw_cycles: float = 15.0
    min_snr_db: float = 30.0
    tolerance_db: float = 2.0
    tolerance_low_db: float = 3.0  # below 125 Hz

    # Reserved: subwoofer on the S3's sub output (default stereo only)
    sub: bool = False

    extra: dict = field(default_factory=dict)


def load_config(path: Path | None = None) -> Config:
    path = path or CONFIG_FILE
    cfg = Config()
    if path.exists():
        data = tomllib.loads(path.read_text())
        known = {f.name for f in fields(Config)}
        for key, value in data.items():
            if key in known:
                if key == "bands":
                    value = tuple(float(v) for v in value)
                setattr(cfg, key, value)
            else:
                cfg.extra[key] = value
    return cfg


def band_label(f: float) -> str:
    if f >= 1000:
        k = f / 1000
        return f"{k:g}k"
    return f"{f:g}"


def load_streamer_model(cfg: Config, path: Path | None = None) -> dict[float, dict]:
    """Per-band filter model: {center: {"fc": Hz, "q": Q, "gain_scale": x}}."""
    path = path or STREAMER_FILE
    model = {f: {"fc": f, "q": cfg.default_q, "gain_scale": 1.0} for f in cfg.bands}
    if path.exists():
        learned = json.loads(path.read_text()).get("bands", {})
        for key, params in learned.items():
            f = float(key)
            if f in model:
                model[f].update(params)
    return model


def save_streamer_band(center: float, params: dict, path: Path | None = None) -> None:
    path = path or STREAMER_FILE
    data = json.loads(path.read_text()) if path.exists() else {"device": "Fosi Audio S3", "bands": {}}
    data["bands"][f"{center:g}"] = params
    path.write_text(json.dumps(data, indent=2))
