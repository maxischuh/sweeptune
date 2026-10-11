"""Microphone profiles and REW-style calibration tables.

Built-in profiles live in the package (`data/mics/*.json`); a file with the same
id in `./mics/` overrides it (this is where `mic relcal` writes derived cals).
Calibration tables are applied by *subtracting* the mic's response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

USER_DIR = Path("mics")


@dataclass
class MicProfile:
    id: str
    name: str
    kind: str
    connection: str
    input_patterns: list[str]
    input_excludes: list[str] = field(default_factory=list)
    os_processing: bool = False
    trust_low_hz: float = 20.0
    trust_high_hz: float = 20000.0
    trust_level: str = "primary"
    placement: str = ""
    notes: list[str] = field(default_factory=list)
    calibration: list[list[float]] | None = None
    cal_file: str | None = None
    approximate: bool = True

    def cal_table(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.cal_file:
            path = Path(self.cal_file)
            if not path.is_absolute():
                path = USER_DIR / path
            if path.exists():
                return parse_rew_cal(path.read_text())
        if self.calibration:
            arr = np.asarray(self.calibration, dtype=float)
            return arr[:, 0], arr[:, 1]
        return None

    def correct(self, freqs: np.ndarray, db: np.ndarray) -> np.ndarray:
        table = self.cal_table()
        if table is None:
            return db
        f, c = table
        return db - np.interp(np.log10(freqs), np.log10(f), c)

    def trust_mask(self, freqs: np.ndarray) -> np.ndarray:
        return (freqs >= self.trust_low_hz) & (freqs <= self.trust_high_hz)


def parse_rew_cal(text: str) -> tuple[np.ndarray, np.ndarray]:
    """REW / UMIK-style text: `freq dB [phase]`, header lines and quotes ignored."""
    f, d = [], []
    for line in text.splitlines():
        parts = line.replace(",", " ").replace(";", " ").split()
        if len(parts) < 2:
            continue
        try:
            fv, dv = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        f.append(fv)
        d.append(dv)
    if len(f) < 2:
        raise ValueError("calibration file has fewer than two `freq dB` rows")
    order = np.argsort(f)
    return np.asarray(f)[order], np.asarray(d)[order]


def write_rew_cal(path: Path, freqs: np.ndarray, db: np.ndarray, header: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'"{header}"'] if header else []
    lines += [f"{f:.2f}\t{d:.2f}" for f, d in zip(freqs, db)]
    path.write_text("\n".join(lines) + "\n")
    return path


def _builtin_dir():
    return resources.files("sweeptune") / "data" / "mics"


def list_profiles() -> list[MicProfile]:
    ids = {p.name[:-5] for p in _builtin_dir().iterdir() if p.name.endswith(".json")}
    if USER_DIR.exists():
        ids |= {p.stem for p in USER_DIR.glob("*.json")}
    return [load_profile(i) for i in sorted(ids)]


def load_profile(mic_id: str) -> MicProfile:
    user = USER_DIR / f"{mic_id}.json"
    if user.exists():
        data = json.loads(user.read_text())
    else:
        res = _builtin_dir() / f"{mic_id}.json"
        if not res.is_file():
            known = ", ".join(p.id for p in list_profiles())
            raise KeyError(f"unknown mic {mic_id!r} (known: {known})")
        data = json.loads(res.read_text())
    known_fields = MicProfile.__dataclass_fields__
    return MicProfile(**{k: v for k, v in data.items() if k in known_fields})


def save_user_profile(profile: MicProfile) -> Path:
    USER_DIR.mkdir(exist_ok=True)
    path = USER_DIR / f"{profile.id}.json"
    path.write_text(json.dumps(profile.__dict__, indent=2, ensure_ascii=False))
    return path


def relative_cal(freqs: np.ndarray, test_db: np.ndarray, ref_db: np.ndarray, norm_hz: float = 1000.0):
    """Derived calibration test/ref, normalized to 0 dB at `norm_hz`."""
    diff = np.asarray(test_db) - np.asarray(ref_db)
    return diff - np.interp(np.log10(norm_hz), np.log10(freqs), diff)
