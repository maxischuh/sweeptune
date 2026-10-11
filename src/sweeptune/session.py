"""Session folders: `sessions/<timestamp>/` with one sub-folder per iteration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("sessions")


@dataclass
class Measurement:
    fs: int
    mic: str
    kind: str  # baseline | verify | probe-before | probe-after | relcal
    applied_eq: list[float]
    irs: dict[str, np.ndarray]  # "p01-L" -> impulse response
    info: dict = field(default_factory=dict)  # per position: latency, snr, levels ...
    level_dbfs: float = 0.0

    def positions(self) -> list[str]:
        return sorted({k.split("-")[0] for k in self.irs})

    def channels(self) -> list[str]:
        return sorted({k.split("-")[1] for k in self.irs})


@dataclass
class Session:
    path: Path
    meta: dict

    @classmethod
    def create(cls, **meta) -> "Session":
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = ROOT / stamp
        path.mkdir(parents=True, exist_ok=False)
        meta = {"created": datetime.now().isoformat(timespec="seconds"), "iterations": [], **meta}
        s = cls(path, meta)
        s.save()
        return s

    @classmethod
    def open(cls, path: Path | str | None = None) -> "Session":
        if path is None:
            if not ROOT.exists() or not any(ROOT.iterdir()):
                raise FileNotFoundError("no sessions yet: run `sweeptune measure` first")
            path = sorted(p for p in ROOT.iterdir() if (p / "session.json").exists())[-1]
        path = Path(path)
        return cls(path, json.loads((path / "session.json").read_text()))

    def save(self) -> None:
        (self.path / "session.json").write_text(json.dumps(self.meta, indent=2))

    # iterations -----------------------------------------------------------
    def new_iteration(self, kind: str) -> Path:
        idx = len(self.meta["iterations"])
        d = self.path / f"iter-{idx:02d}-{kind}"
        d.mkdir()
        self.meta["iterations"].append({"index": idx, "kind": kind, "dir": d.name})
        self.save()
        return d

    def iteration_dirs(self, kinds: tuple[str, ...] = ("baseline", "verify")) -> list[Path]:
        return [self.path / it["dir"] for it in self.meta["iterations"] if it["kind"] in kinds]

    def latest_measurement_dir(self) -> Path:
        dirs = [d for d in self.iteration_dirs() if (d / "measurement.json").exists()]
        if not dirs:
            raise FileNotFoundError(f"no measurement in {self.path}")
        return dirs[-1]

    def last_suggestion(self) -> dict | None:
        for d in reversed(self.iteration_dirs()):
            f = d / "suggestion.json"
            if f.exists():
                return json.loads(f.read_text())
        return None


def save_measurement(d: Path, m: Measurement, recordings: dict[str, np.ndarray] | None = None) -> None:
    np.savez_compressed(d / "irs.npz", **m.irs)
    meta = {k: v for k, v in m.__dict__.items() if k != "irs"}
    (d / "measurement.json").write_text(json.dumps(meta, indent=2, default=_json_default))
    for name, rec in (recordings or {}).items():
        sf.write(d / f"{name}.wav", np.clip(rec, -1, 1), m.fs, subtype="PCM_24")


def load_measurement(d: Path) -> Measurement:
    meta = json.loads((d / "measurement.json").read_text())
    with np.load(d / "irs.npz") as z:
        irs = {k: z[k] for k in z.files}
    return Measurement(irs=irs, **meta)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))
