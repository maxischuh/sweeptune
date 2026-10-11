"""CoreAudio I/O via sounddevice: device matching and simultaneous play/record.

Playback (AirPlay to the Fosi S3) and capture (the measurement mic) run on two
independent streams, because they are different devices with different clocks;
alignment is done afterwards from the deconvolved impulse response.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy import signal


def _sd():
    import sounddevice as sd  # imported lazily so tests/simulation never need PortAudio

    return sd


def refresh_devices() -> None:
    """Re-scan CoreAudio (AirPlay devices appear once selected in macOS)."""
    sd = _sd()
    sd._terminate()
    sd._initialize()


def list_devices() -> list[dict]:
    sd = _sd()
    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append({"index": i, "name": d["name"], "inputs": d["max_input_channels"],
                    "outputs": d["max_output_channels"], "rate": d["default_samplerate"]})
    return out


def default_devices() -> tuple[int | None, int | None]:
    sd = _sd()
    i, o = sd.default.device
    return (i if i is not None and i >= 0 else None, o if o is not None and o >= 0 else None)


def find_output(fragment: str) -> tuple[int | None, str, str]:
    """(index, name, how): config fragment, else any AirPlay device, else system default."""
    devs = [d for d in list_devices() if d["outputs"] > 0]
    for frag, how in ((fragment, "matched config"), ("AirPlay", "AirPlay fallback")):
        if not frag:
            continue
        for d in devs:
            if frag.lower() in d["name"].lower():
                return d["index"], d["name"], how
    _, o = default_devices()
    if o is None:
        return None, "", "no output device"
    return o, list_devices()[o]["name"], "system default"


def find_input(patterns: list[str], excludes: list[str] | None = None) -> tuple[int | None, str]:
    excludes = [e.lower() for e in (excludes or [])]
    for frag in patterns:
        for d in list_devices():
            name = d["name"].lower()
            if d["inputs"] > 0 and frag.lower() in name and not any(e in name for e in excludes):
                return d["index"], d["name"]
    return None, ""


def supports_rate(device: int, fs: int, kind: str) -> bool:
    sd = _sd()
    try:
        if kind == "input":
            sd.check_input_settings(device=device, samplerate=fs, channels=1)
        else:
            sd.check_output_settings(device=device, samplerate=fs, channels=2)
        return True
    except Exception:
        return False


def _resample(x: np.ndarray, fs_from: int, fs_to: int) -> np.ndarray:
    if fs_from == fs_to:
        return x
    g = gcd(fs_from, fs_to)
    return signal.resample_poly(x, fs_to // g, fs_from // g).astype(np.float32)


@dataclass
class LiveBackend:
    out_dev: int
    in_dev: int
    output_name: str
    input_name: str
    preferred_fs: int = 48000
    max_latency_s: float = 5.0

    @property
    def fs(self) -> int:
        """Playback rate: preferred if the output accepts it, else its default."""
        if supports_rate(self.out_dev, self.preferred_fs, "output"):
            return self.preferred_fs
        return int(list_devices()[self.out_dev]["rate"])

    def _in_fs(self, fs: int) -> int:
        if supports_rate(self.in_dev, fs, "input"):
            return fs
        return int(list_devices()[self.in_dev]["rate"])

    def play_and_record(self, sig: np.ndarray, fs: int) -> np.ndarray:
        sd = _sd()
        fs_in = self._in_fs(fs)
        n_rec = int((len(sig) / fs + self.max_latency_s) * fs_in)
        rec = np.zeros(n_rec, dtype=np.float32)
        state = {"r": 0, "p": 0, "xruns": 0}
        done = threading.Event()
        out_ch = min(2, list_devices()[self.out_dev]["outputs"])
        play = sig[:, :out_ch] if out_ch < 2 else sig

        def in_cb(indata, frames, _t, status):
            if status:
                state["xruns"] += 1
            r = state["r"]
            k = min(frames, n_rec - r)
            rec[r: r + k] = indata[:k, 0]
            state["r"] += k
            if state["r"] >= n_rec:
                raise sd.CallbackStop

        def out_cb(outdata, frames, _t, status):
            p = state["p"]
            chunk = play[p: p + frames]
            outdata[: len(chunk)] = chunk
            outdata[len(chunk):] = 0
            state["p"] += frames
            if state["p"] >= len(play):
                raise sd.CallbackStop

        ins = sd.InputStream(device=self.in_dev, samplerate=fs_in, channels=1, dtype="float32",
                             callback=in_cb, finished_callback=done.set)
        outs = sd.OutputStream(device=self.out_dev, samplerate=fs, channels=out_ch, dtype="float32",
                               callback=out_cb)
        with ins, outs:
            if not done.wait(timeout=n_rec / fs_in + 15):
                raise TimeoutError("recording did not finish (input device stalled?)")
        if state["xruns"]:
            print(f"warning: {state['xruns']} input over/underflows during recording")
        return _resample(rec, fs_in, fs)

    def record(self, seconds: float, fs: int) -> np.ndarray:
        sd = _sd()
        fs_in = self._in_fs(fs)
        x = sd.rec(int(seconds * fs_in), samplerate=fs_in, channels=1, dtype="float32", device=self.in_dev)
        sd.wait()
        return _resample(x[:, 0], fs_in, fs)
