"""sweeptune command-line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from . import audio_io
from . import balance as bal
from .analysis import FREQS, band_levels, fdw_response, level_offset, power_average, smooth
from .config import Config, band_label, load_config, load_streamer_model, save_streamer_band
from .eq import eq_curve, fit_probe, format_eq, parse_eq
from .measure import (MeasurementError, Suggestion, acoustics_summary, agc_check, analyze_measurement,
                      level_probe, low_snr_bands, measure_positions, pool, process_recording, suggest)
from .mics import MicProfile, list_profiles, load_profile, relative_cal, save_user_profile, write_rew_cal, USER_DIR
from .session import Measurement, Session, load_measurement, save_json, save_measurement
from .simulate import SimBackend
from .sweep import SweepPlan, export_wav
from .targets import tilt_advice

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Measure your room with a sweep and get settings for the Fosi S3 10-band EQ.")
mic_app = typer.Typer(no_args_is_help=True, help="Microphone profiles and relative calibration.")
eq_app = typer.Typer(no_args_is_help=True, help="EQ suggestions and filter probing.")
sweep_app = typer.Typer(no_args_is_help=True, help="Sweep files for offline recording.")
app.add_typer(mic_app, name="mic")
app.add_typer(eq_app, name="eq")
app.add_typer(sweep_app, name="sweep")

console = Console()


class State:
    simulate = False
    yes = False
    cfg = Config()


state = State()


@app.callback()
def main(simulate: bool = typer.Option(False, "--simulate", help="Use a synthetic room instead of real audio."),
         yes: bool = typer.Option(False, "--yes", "-y", help="Answer yes to all prompts."),
         config: Optional[Path] = typer.Option(None, help="Path to sweeptune.toml.")):
    state.simulate = simulate
    state.yes = yes or simulate
    state.cfg = load_config(config)


# ---------------------------------------------------------------------- helpers
def ask(msg: str) -> bool:
    if state.yes:
        console.print(f"[dim]{msg} → yes[/]")
        return True
    return typer.confirm(msg, default=True)


def fail(msg: str) -> None:
    console.print(f"[bold red]Error:[/] {msg}")
    raise typer.Exit(1)


def make_backend(mic: MicProfile, applied_eq):
    cfg = state.cfg
    if state.simulate:
        return SimBackend(mic=mic.id, applied_eq=np.asarray(applied_eq, dtype=float),
                          max_latency_s=cfg.max_latency_s)
    audio_io.refresh_devices()
    out_idx, out_name, how = audio_io.find_output(cfg.output_device)
    if out_idx is None:
        fail("no audio output device found")
    if how != "matched config":
        console.print(f"[yellow]Output '{cfg.output_device}' not found; using {how}: {out_name}.[/] "
                      "Select the Fosi S3 under Control Center → Sound (AirPlay) first.")
    patterns = [cfg.input_device] if cfg.input_device else mic.input_patterns
    in_idx, in_name = audio_io.find_input(patterns, [] if cfg.input_device else mic.input_excludes)
    if in_idx is None:
        hint = (" Plug-in check: the MacBook jack is TRRS; a TS/TRS mic plug may not register — "
                "use a TRS→TRRS mic adapter.") if mic.id == "audyssey" else ""
        fail(f"input for mic '{mic.id}' not found (patterns {patterns}).{hint} Run `sweeptune doctor`.")
    return audio_io.LiveBackend(out_idx, in_idx, out_name, in_name, cfg.sample_rate, cfg.max_latency_s)


def set_position(backend):
    return (lambda p: setattr(backend, "position", p)) if isinstance(backend, SimBackend) else None


def choose_level(backend, start: float) -> float:
    cfg = state.cfg
    console.print(Panel("Set the S3 volume to a moderate listening level. The first test sweep plays at "
                        f"{start:.0f} dBFS; you can step it up in 6 dB steps.", title="Level check"))
    level = start
    while True:
        r = level_probe(backend, cfg, level)
        console.print(f"  sweep {level:+.0f} dBFS → mic peak {r['peak_dbfs']:.1f} dBFS, "
                      f"SNR ~{r['snr_db']:.0f} dB, latency {r['latency_s'] * 1000:.0f} ms")
        if r["clipped"]:
            console.print("[red]  input clipped, stepping down 6 dB[/]")
            level -= 6
            continue
        if state.yes:
            if (r["snr_db"] < 45 or r["peak_dbfs"] < -30) and level + 6 <= cfg.max_level_dbfs:
                level += 6
                continue
            return level
        choice = Prompt.ask("  [a]ccept, [u]p 6 dB, [d]own 6 dB", choices=["a", "u", "d"], default="a")
        if choice == "a":
            return level
        if choice == "u":
            if level + 6 > cfg.max_level_dbfs:
                console.print(f"  already near the {cfg.max_level_dbfs} dBFS limit: raise the S3 volume instead")
            else:
                level += 6
        else:
            level -= 6


def check_agc(backend, mic: MicProfile, level: float) -> None:
    if not mic.os_processing:
        return
    ok, diff = agc_check(backend, state.cfg, level)
    if ok:
        console.print(f"[green]AGC check ok[/] (6 dB step measured as {diff:.1f} dB)")
        return
    console.print(f"[yellow]AGC/voice processing suspected:[/] a 6 dB level step measured as {diff:.1f} dB. "
                  "Disable Voice Isolation (Control Center → Mic Mode → Standard) or use the offline path "
                  "(`sweeptune sweep export` + `sweeptune analyze`).")
    if not ask("Continue anyway?"):
        raise typer.Exit(1)


def print_measurement(m: Measurement) -> None:
    t = Table(title=f"{m.kind} measurement ({m.mic})")
    for col in ("Position", "Latency", "Arrival vs first ch.", "Input peak", "Min SNR", "Warnings"):
        t.add_column(col)
    for pos, info in m.info.items():
        t.add_row(pos, f"{info['latency_s'] * 1000:.0f} ms", ", ".join(f"{ch} {v:+.2f} ms" for ch, v in info["arrival_ms"].items()),
                  f"{info['input_peak_dbfs']:.1f} dBFS", f"{min(info['snr_db'].values()):.0f} dB",
                  "; ".join(info.get("warnings", [])))
    console.print(t)


def print_balance(b: bal.Balance, title: str) -> None:
    t = Table(title=title)
    t.add_column("Octave")
    t.add_column("R − L", justify="right")
    for band, v in b.per_octave.items():
        mark = "" if 500 <= band <= 4000 else " [dim](room/directivity, not balance)[/]"
        t.add_row(f"{band_label(band)} Hz", f"{v:+.1f} dB{mark}")
    console.print(t)
    spread = "L repeatability" if b.distance_cm is not None else f"spread over {b.positions} positions"
    console.print(f"Balance (500 Hz–4 kHz): [bold]R − L = {b.r_minus_l_db:+.1f} dB[/] ({spread} ±{b.spread_db:.1f} dB)")
    console.print(b.advice(state.cfg.eq_step_db))
    if b.distance_advice():
        console.print(b.distance_advice())
    if b.distance_cm is None and b.spread_db > 1.5:
        console.print("[yellow]Large spread between positions:[/] confirm at the seat with `sweeptune balance --live`.")


def balance_hint(m: Measurement) -> None:
    b = bal.from_measurement(m.irs, m.fs)
    if b is not None:
        console.print(f"Channel balance: R − L {b.r_minus_l_db:+.1f} dB → `sweeptune balance` for details.")


def fmt_db(v: float) -> str:
    return f"{v:+.1f} dB" if np.isfinite(v) else "n/a"


def print_suggestion(sg: Suggestion) -> None:
    cfg = state.cfg
    r = sg.result
    t = Table(title=f"Fosi S3 EQ — target {sg.target_name}")
    for col in ("Band", "Now", "Set to", "Δ", "Expected residual", "Note"):
        t.add_column(col, justify="right" if col not in ("Band", "Note") else "left")
    for i, c in enumerate(cfg.bands):
        d = r.gains[i] - sg.current[i]
        t.add_row(f"{band_label(c)} Hz", f"{sg.current[i]:+.1f}", f"[bold]{r.gains[i]:+.1f}[/]",
                  f"{d:+.1f}" if d else "·", fmt_db(r.residual_bands[i]), sg.limits.reasons.get(c, ""))
    console.print(t)
    console.print(Panel(r.copy_string(), title="Copy string", expand=False))
    if r.headroom_db > 0:
        console.print(f"[yellow]Headroom:[/] the EQ boosts up to {r.headroom_db:.1f} dB → "
                      f"lower the S3 volume by {r.headroom_db:.1f} dB to avoid clipping.")
    if sg.f_low > FREQS[0] * 1.01:
        console.print(f"Speaker bass extension (−6 dB vs target): ~{sg.f_low:.0f} Hz — "
                      "the target follows the natural roll-off below that.")


def suggestion_dict(sg: Suggestion, ra) -> dict:
    r = sg.result
    return {
        "target": sg.target_name, "gains": r.gains, "raw": r.raw, "residual_bands": r.residual_bands,
        "headroom_db": r.headroom_db, "offset": r.offset, "f_low": sg.f_low, "current": sg.current,
        "reasons": {f"{c:g}": v for c, v in sg.limits.reasons.items()},
        "room": ra.room, "target_curve": sg.target, "target_adapted": sg.target_adapted,
    }


def channels_opt(text: str) -> tuple[str, ...]:
    return tuple(c.strip().upper() for c in text.split(",") if c.strip())


def run_suggest(s: Session, d: Path, m: Measurement, target: Optional[str] = None, quiet: bool = False):
    cfg = state.cfg
    if any(i.get("clipped") for i in m.info.values()):
        fail("this measurement contains clipped recordings; re-measure at a lower level")
    mic = load_profile(m.mic)
    model = load_streamer_model(cfg)
    ra = analyze_measurement(m, mic, cfg, model)
    dirs = s.iteration_dirs()
    earlier = dirs[: dirs.index(d)] if d in dirs else []
    others = [analyze_measurement(load_measurement(o), mic, cfg, model) for o in earlier
              if (o / "measurement.json").exists()]
    ra = pool(ra, others)
    sg = suggest(ra, mic, cfg, model, np.asarray(m.applied_eq), target or s.meta.get("target"),
                 low_snr_bands(m, cfg))
    save_json(d / "suggestion.json", suggestion_dict(sg, ra))
    if not quiet:
        print_suggestion(sg)
    return sg, ra, mic, model


# ---------------------------------------------------------------------- doctor
@app.command()
def doctor(mic: str = typer.Option("audyssey", help="Mic profile whose input to test.")):
    """Check audio routing, mic inputs and the noise floor."""
    cfg = state.cfg
    if state.simulate:
        console.print("[green]Simulation mode:[/] synthetic Fosi S3 over AirPlay + synthetic room; no audio devices used.")
        return
    audio_io.refresh_devices()
    t = Table(title="CoreAudio devices")
    for col in ("#", "Name", "In", "Out", "Rate"):
        t.add_column(col)
    for d in audio_io.list_devices():
        t.add_row(str(d["index"]), d["name"], str(d["inputs"]), str(d["outputs"]), f"{d['rate']:.0f}")
    console.print(t)

    out_idx, out_name, how = audio_io.find_output(cfg.output_device)
    if how == "matched config":
        console.print(f"[green]✓ Output:[/] {out_name}")
    else:
        console.print(f"[yellow]! Output:[/] '{cfg.output_device}' not found, would use {how}: {out_name}. "
                      "Select the Fosi S3 in Control Center → Sound (AirPlay), then re-run doctor. "
                      "(Or set `output_device` in sweeptune.toml.)")
    if out_idx is not None and not audio_io.supports_rate(out_idx, cfg.sample_rate, "output"):
        console.print(f"  output runs at {audio_io.list_devices()[out_idx]['rate']:.0f} Hz; sweeps will use that rate.")

    for p in list_profiles():
        idx, name = audio_io.find_input(p.input_patterns, p.input_excludes)
        if idx is not None:
            console.print(f"[green]✓ Mic '{p.id}':[/] {name}")
        else:
            extra = ""
            if p.id == "audyssey":
                extra = (" — no external mic detected. The MacBook jack is CTIA TRRS (mic on the sleeve); "
                         "a TS/TRS plug shorts it and macOS sees headphones only. Use a TRS→TRRS mic adapter "
                         "(~5 €) or a USB-C interface with plug-in power.")
            elif p.id == "iphone17pro":
                extra = " — unlock the iPhone near the Mac (same Apple ID, Wi-Fi + Bluetooth on)."
            console.print(f"[yellow]✗ Mic '{p.id}' not found[/]{extra}")

    profile = load_profile(mic)
    idx, name = audio_io.find_input(profile.input_patterns, profile.input_excludes)
    if idx is None:
        return
    console.print(f"\nRecording 2 s of silence on {name} …")
    backend = audio_io.LiveBackend(out_idx or 0, idx, out_name, name, cfg.sample_rate)
    x = backend.record(2.0, cfg.sample_rate)
    rms = 20 * np.log10(max(np.sqrt(np.mean(x**2)), 1e-12))
    console.print(f"Noise floor: {rms:.1f} dBFS RMS")
    if rms < -100:
        console.print("[red]Input is digitally silent: the mic is not powered or the plug shorts the mic "
                      "contact (see TRRS note above).[/]")
    elif rms > -45:
        console.print("[yellow]Noise floor is high: quiet the room (fans, fridge) or check the cable.[/]")
    else:
        console.print("[green]Noise floor looks sane.[/]")


# ---------------------------------------------------------------------- mic
@mic_app.command("list")
def mic_list():
    """List microphone profiles."""
    t = Table()
    for col in ("id", "name", "trust", "range", "cal"):
        t.add_column(col)
    for p in list_profiles():
        t.add_row(p.id, p.name, p.trust_level, f"{p.trust_low_hz:g}–{p.trust_high_hz:g} Hz",
                  "yes" if p.cal_table() is not None else "none")
    console.print(t)


@mic_app.command("show")
def mic_show(mic_id: str):
    """Show one microphone profile."""
    p = load_profile(mic_id)
    console.print_json(json.dumps(p.__dict__, ensure_ascii=False))


@mic_app.command("relcal")
def mic_relcal(test: str, ref: str = typer.Option("audyssey", "--ref"),
               channel: str = typer.Option("L", help="Speaker to measure in the near field."),
               level: Optional[float] = typer.Option(None, help="Sweep level dBFS (skip the level check).")):
    """Derive a calibration for TEST relative to REF at the same near-field spot."""
    cfg = state.cfg
    console.print(Panel(f"Place the mic ~30 cm in front of the {channel} speaker's tweeter, on axis. "
                        "Mark the spot: both mics must sit at exactly the same place.", title="Relative calibration"))
    curves = {}
    for which in (ref, test):
        prof = load_profile(which)
        backend = make_backend(prof, np.zeros(len(cfg.bands)))
        plan = SweepPlan.from_config(cfg, fs=backend.fs, channels=(channel,))
        if not ask(f"Put the {prof.name} at the marked spot. Ready?"):
            raise typer.Exit(1)
        lvl = level if level is not None else choose_level(backend, cfg.start_level_dbfs)
        if isinstance(backend, SimBackend):
            backend.position = 1
        rec = backend.play_and_record(plan.signal(lvl), plan.fs)
        irs, _ = process_recording(rec, plan, cfg)
        db = fdw_response(irs[channel], plan.fs, cycles=cfg.fdw_cycles)
        if which == ref:
            db = prof.correct(FREQS, db)
        curves[which] = smooth(FREQS, db, 1 / 6)
    cal = relative_cal(FREQS, curves[test], curves[ref])
    cal_path = write_rew_cal(USER_DIR / f"{test}.cal.txt", FREQS, cal, header=f"{test} relative to {ref}")
    prof = load_profile(test)
    prof.cal_file = cal_path.name
    prof.notes = prof.notes + [f"Relative calibration vs '{ref}' written by sweeptune mic relcal."]
    save_user_profile(prof)
    t = Table(title=f"{test} relative to {ref} (0 dB @ 1 kHz)")
    t.add_column("Band")
    t.add_column("dB", justify="right")
    for c, v in zip(cfg.bands, band_levels(FREQS, cal, cfg.bands)):
        t.add_row(f"{band_label(c)} Hz", f"{v:+.1f}")
    console.print(t)
    console.print(f"Saved {cal_path} and {USER_DIR / (test + '.json')}")


# ---------------------------------------------------------------------- sweep
@sweep_app.command("export")
def sweep_export(out: Path = typer.Option(Path("sweep.wav")),
                 channels: str = typer.Option("L,R"),
                 level: float = typer.Option(-12.0, help="Sweep peak level in dBFS.")):
    """Write the measurement sweep as a WAV (record it on the iPhone, then `sweeptune analyze`)."""
    plan = SweepPlan.from_config(state.cfg, channels=channels_opt(channels))
    export_wav(plan, out, level)
    console.print(f"Wrote {out} ({plan.duration_s():.1f} s, {plan.fs} Hz, channels {','.join(plan.channels)}).")
    console.print("Start recording on the iPhone first (measurement mode / no processing, 48 kHz WAV), "
                  "then play the file on the S3. One recording per mic position.")


# ---------------------------------------------------------------------- measure
@app.command()
def measure(mic: str = typer.Option("audyssey", help="Mic profile id."),
            positions: int = typer.Option(7, help="Mic positions around the seat."),
            channels: str = typer.Option("L,R", help="L, R and/or LR (both speakers)."),
            eq: str = typer.Option("flat", help="EQ currently set on the S3, e.g. '31.5:-3 63:-4'."),
            level: Optional[float] = typer.Option(None, help="Sweep level dBFS (skip the level check)."),
            target: Optional[str] = typer.Option(None, help="Target curve for this session.")):
    """Baseline: guided multi-position sweep measurement (starts a new session)."""
    cfg = state.cfg
    profile = load_profile(mic)
    applied = parse_eq(eq, cfg.bands)
    backend = make_backend(profile, applied)
    console.print(f"Output: [bold]{backend.output_name}[/] · Input: [bold]{backend.input_name}[/]")
    plan = SweepPlan.from_config(cfg, fs=backend.fs, channels=channels_opt(channels))
    console.print(f"Placement: {profile.placement}")
    lvl = level if level is not None else choose_level(backend, cfg.start_level_dbfs)
    check_agc(backend, profile, lvl)
    s = Session.create(mic=mic, output=backend.output_name, input=backend.input_name,
                       target=target or cfg.target, level_dbfs=lvl, channels=list(plan.channels),
                       simulate=state.simulate)
    d = s.new_iteration("baseline")
    try:
        irs, infos, recs = measure_positions(backend, cfg, plan, lvl, positions, ask, set_position(backend),
                                             (profile.trust_low_hz, profile.trust_high_hz))
    except MeasurementError as e:
        fail(str(e))
    m = Measurement(fs=plan.fs, mic=mic, kind="baseline", applied_eq=list(applied), irs=irs, info=infos,
                    level_dbfs=lvl)
    save_measurement(d, m, recs)
    print_measurement(m)
    balance_hint(m)
    console.print(f"Saved to {d}. Next: [bold]sweeptune eq suggest[/]")


@app.command()
def analyze(wavs: list[Path] = typer.Argument(..., help="One recording per mic position."),
            mic: str = typer.Option("iphone17pro"),
            channels: str = typer.Option("L,R", help="Must match the exported sweep."),
            eq: str = typer.Option("flat", help="EQ that was set on the S3 while recording."),
            append: bool = typer.Option(False, help="Add as a verify iteration to the latest session.")):
    """Analyze offline recordings of `sweeptune sweep export`."""
    from scipy.signal import resample_poly

    cfg = state.cfg
    profile = load_profile(mic)
    applied = parse_eq(eq, cfg.bands)
    plan = SweepPlan.from_config(cfg, channels=channels_opt(channels))
    irs, infos = {}, {}
    for i, path in enumerate(wavs, start=1):
        data, fs = sf.read(path, always_2d=True)
        x = data[:, 0].astype(np.float32)
        if fs != plan.fs:
            from math import gcd

            g = gcd(fs, plan.fs)
            x = resample_poly(x, plan.fs // g, fs // g).astype(np.float32)
        try:
            ch_irs, info = process_recording(x, plan, cfg)
        except RuntimeError as e:
            fail(f"{path}: {e}")
        tag = f"p{i:02d}"
        irs.update({f"{tag}-{ch}": ir for ch, ir in ch_irs.items()})
        infos[tag] = info
    if append:
        s = Session.open()
        d = s.new_iteration("verify")
        kind = "verify"
    else:
        s = Session.create(mic=mic, output="offline WAV", input=str(wavs[0]), target=cfg.target,
                           level_dbfs=0.0, channels=list(plan.channels), simulate=False)
        d = s.new_iteration("baseline")
        kind = "baseline"
    m = Measurement(fs=plan.fs, mic=mic, kind=kind, applied_eq=list(applied), irs=irs, info=infos)
    save_measurement(d, m)
    print_measurement(m)
    balance_hint(m)
    console.print(f"Saved to {d}. Next: [bold]sweeptune eq suggest[/]")


# ---------------------------------------------------------------------- eq
@eq_app.command("suggest")
def eq_suggest(target: Optional[str] = typer.Option(None, help="harman-like | bk-1974 | flat | file.csv"),
               max_boost: Optional[float] = typer.Option(None, help="Max boost per band (dB)."),
               shelf_db: Optional[float] = typer.Option(None, help="Bass shelf for harman-like (dB)."),
               session: Optional[Path] = typer.Option(None)):
    """Suggest S3 EQ settings from the latest measurement."""
    cfg = state.cfg
    if max_boost is not None:
        cfg.max_boost_db = max_boost
    if shelf_db is not None:
        cfg.shelf_db = shelf_db
    try:
        s = Session.open(session)
        d = s.latest_measurement_dir()
    except FileNotFoundError as e:
        fail(str(e))
    m = load_measurement(d)
    sg, ra, mic, _ = run_suggest(s, d, m, target)
    ac = acoustics_summary(m, ra)
    advice = tilt_advice(ac["rt60_mid"], cfg)
    if advice:
        console.print(f"[yellow]Note:[/] {advice}")
    console.print("Enter the values on the S3, then run [bold]sweeptune verify[/].")


@eq_app.command("probe")
def eq_probe(band: str = typer.Argument(..., help="Band to probe, e.g. 125 or 1k."),
             delta: float = typer.Option(-6.0, help="Temporary change to apply (dB)."),
             positions: int = typer.Option(1)):
    """Learn the S3's real filter shape for one band from a before/after measurement."""
    cfg = state.cfg
    center = cfg.bands[int(np.argmax(parse_eq(f"{band}:1", cfg.bands)))]
    try:
        s = Session.open()
        prev = load_measurement(s.latest_measurement_dir())
    except FileNotFoundError as e:
        fail(str(e))
    current = np.asarray(prev.applied_eq, dtype=float)
    i = cfg.bands.index(center)
    probe_eq = current.copy()
    probe_eq[i] = np.clip(current[i] + delta, cfg.eq_min_db, cfg.eq_max_db)
    delta = probe_eq[i] - current[i]
    profile = load_profile(prev.mic)
    backend = make_backend(profile, current)
    plan = SweepPlan.from_config(cfg, fs=backend.fs, channels=tuple(s.meta.get("channels", ["L", "R"])))
    lvl = s.meta.get("level_dbfs", cfg.start_level_dbfs)
    d = s.new_iteration("probe")
    curves = []
    for step, eq_now in (("before", current), ("after", probe_eq)):
        if step == "after":
            if not ask(f"Set the {band_label(center)} Hz band to {probe_eq[i]:+.1f} dB on the S3 "
                       "(don't move the mic). Done?"):
                raise typer.Exit(1)
            if isinstance(backend, SimBackend):
                backend.applied_eq = probe_eq
        try:
            irs, infos, _ = measure_positions(backend, cfg, plan, lvl, positions, ask, set_position(backend),
                                              (profile.trust_low_hz, profile.trust_high_hz))
        except MeasurementError as e:
            fail(str(e))
        m = Measurement(fs=plan.fs, mic=prev.mic, kind=f"probe-{step}", applied_eq=list(eq_now), irs=irs, info=infos)
        if step == "after":
            save_measurement(d, m)
        curves.append(power_average([fdw_response(ir, plan.fs, cycles=cfg.fdw_cycles) for ir in irs.values()]))
    diff = smooth(FREQS, curves[1] - curves[0], 1 / 12)
    fit = fit_probe(FREQS, diff, center, delta)
    save_streamer_band(center, {k: fit[k] for k in ("fc", "q", "gain_scale")})
    save_json(d / "probe.json", {"band": center, "delta": delta, **fit})
    console.print(f"{band_label(center)} Hz band: fc {fit['fc']:.1f} Hz, Q {fit['q']:.2f}, "
                  f"gain scale {fit['gain_scale']:.2f} (fit error {fit['fit_rms_db']:.2f} dB rms) → saved to streamer.json")
    console.print(f"[bold]Set the {band_label(center)} Hz band back to {current[i]:+.1f} dB.[/]")


# ---------------------------------------------------------------------- balance
@app.command()
def balance(live: bool = typer.Option(False, "--live", help="Re-measure at the seat while you turn the knob."),
            mic: str = typer.Option("audyssey", help="Mic profile (live mode)."),
            level: Optional[float] = typer.Option(None, help="Sweep level dBFS (live mode, skips the level check)."),
            session: Optional[Path] = typer.Option(None, help="Session to read (default: latest).")):
    """Left/right balance: how far to turn the amp's balance control."""
    cfg = state.cfg
    if not live:
        try:
            s = Session.open(session)
            m = load_measurement(s.latest_measurement_dir())
        except FileNotFoundError as e:
            fail(f"{e} (or use `sweeptune balance --live`)")
        b = bal.from_measurement(m.irs, m.fs)
        if b is None:
            fail("the latest measurement has no separate L and R sweeps; measure with --channels L,R")
        if any(m.applied_eq):
            console.print("[dim]The S3 EQ applies to both channels equally, so it does not affect the balance.[/]")
        print_balance(b, f"Channel balance from {s.path.name} ({b.positions} positions)")
        return

    profile = load_profile(mic)
    backend = make_backend(profile, np.zeros(len(cfg.bands)))
    plan = SweepPlan.from_config(cfg, fs=backend.fs, channels=("L", "R", "L"), n=2**17)
    plan.lead, plan.gap = int(1.0 * plan.fs), int(1.5 * plan.fs)
    console.print(Panel("Put the mic at the listening seat, ear height, centred between the speakers "
                        f"({profile.placement}). Set the amp's balance to centre to start.", title="Balance"))
    lvl = level if level is not None else choose_level(backend, cfg.start_level_dbfs)
    check_agc(backend, profile, lvl)
    while True:
        rec = backend.play_and_record(plan.signal(lvl), plan.fs)
        try:
            irs, info = process_recording(rec, plan, cfg)
        except RuntimeError as e:
            console.print(f"[red]{e}[/]")
            if state.yes or not ask("Try again?"):
                raise typer.Exit(1)
            continue
        if info["clipped"]:
            console.print("[red]Input clipped: lower the level.[/]")
            raise typer.Exit(1)
        b = bal.from_live(irs, info, plan.fs)
        print_balance(b, "Channel balance at the seat (L, R, L sweeps)")
        if b.spread_db > 0.5:
            console.print(f"[yellow]The two L sweeps differ by {b.spread_db:.1f} dB:[/] noise, AirPlay dropout or "
                          "AGC — treat this reading with care.")
        if state.yes or abs(b.r_minus_l_db) <= bal.TOLERANCE_DB:
            break
        if Prompt.ask("Turn the balance knob, then press Enter to re-measure (q to quit)", default="") == "q":
            break


# ---------------------------------------------------------------------- verify
@app.command()
def verify(positions: int = typer.Option(3, help="Mic positions to re-measure."),
           entered: Optional[str] = typer.Option(None, help="Values you set on the S3 (default: the suggestion)."),
           level: Optional[float] = typer.Option(None)):
    """Re-measure with the EQ applied, compare to the prediction and propose the next correction."""
    cfg = state.cfg
    try:
        s = Session.open()
        prev_dir = s.latest_measurement_dir()
    except FileNotFoundError as e:
        fail(str(e))
    sugg_file = prev_dir / "suggestion.json"
    if not sugg_file.exists():
        fail("no suggestion for the latest measurement: run `sweeptune eq suggest` first")
    sugg = json.loads(sugg_file.read_text())
    default = format_eq(cfg.bands, sugg["gains"])
    if entered is None and not state.yes:
        entered = Prompt.ask("Values you set on the S3", default=default)
    applied = parse_eq(entered or default, cfg.bands)
    profile = load_profile(s.meta["mic"])
    backend = make_backend(profile, applied)
    plan = SweepPlan.from_config(cfg, fs=backend.fs, channels=tuple(s.meta.get("channels", ["L", "R"])))
    lvl = level if level is not None else s.meta.get("level_dbfs", cfg.start_level_dbfs)
    d = s.new_iteration("verify")
    try:
        irs, infos, recs = measure_positions(backend, cfg, plan, lvl, positions, ask, set_position(backend),
                                             (profile.trust_low_hz, profile.trust_high_hz))
    except MeasurementError as e:
        fail(str(e))
    m = Measurement(fs=plan.fs, mic=s.meta["mic"], kind="verify", applied_eq=list(applied), irs=irs, info=infos,
                    level_dbfs=lvl)
    save_measurement(d, m, recs)
    print_measurement(m)

    sg, ra, mic, model = run_suggest(s, d, m, quiet=True)
    predicted = np.asarray(sugg["room"]) + eq_curve(FREQS, applied, cfg.bands, model)
    target = np.asarray(sugg["target_adapted"])
    meas = smooth(FREQS, ra.average, 1 / 3)
    pred = smooth(FREQS, predicted, 1 / 3)
    meas_t = meas + level_offset(FREQS, meas, target)
    pred_m = pred + level_offset(FREQS, pred, meas)
    vs_pred = band_levels(FREQS, meas, cfg.bands) - band_levels(FREQS, pred_m, cfg.bands)
    vs_target = band_levels(FREQS, meas_t, cfg.bands) - band_levels(FREQS, target, cfg.bands)

    t = Table(title="Verify: measured vs predicted and vs target")
    for col in ("Band", "Set", "vs predicted", "vs target", "Next", "Δ", "Note"):
        t.add_column(col, justify="left" if col in ("Band", "Note") else "right")
    converged_bands = []
    for i, c in enumerate(cfg.bands):
        tol = cfg.tolerance_low_db if c < 125 else cfg.tolerance_db
        constrained = c in sg.limits.reasons
        ok = abs(vs_target[i]) <= tol or constrained
        converged_bands.append(ok)
        flag = abs(vs_pred[i]) > 1.5
        note = "[yellow]off prediction: entry typo or Q model? try `eq probe`[/]" if flag else ""
        note = note or sg.limits.reasons.get(c, "")
        nxt = sg.result.gains[i]
        t.add_row(f"{band_label(c)} Hz", f"{applied[i]:+.1f}",
                  f"[{'yellow' if flag else 'green'}]{vs_pred[i]:+.1f}[/]",
                  "n/a" if not (mic.trust_low_hz <= c <= mic.trust_high_hz) else
                  f"[{'green' if abs(vs_target[i]) <= tol else 'red'}]{vs_target[i]:+.1f}[/]",
                  f"{nxt:+.1f}", f"{nxt - applied[i]:+.1f}" if nxt != applied[i] else "·", note)
    console.print(t)
    stable = np.max(np.abs(sg.result.gains - applied)) <= cfg.eq_step_db
    if all(converged_bands) or stable:
        why = "all bands within tolerance" if all(converged_bands) else "gains stopped changing"
        final = json.loads((d / "suggestion.json").read_text())
        final["gains"], final["converged"] = list(applied), True
        save_json(d / "suggestion.json", final)
        console.print(f"[bold green]Converged ({why}).[/] Final settings: {format_eq(cfg.bands, applied)}  "
                      "→ run [bold]sweeptune report[/].")
    else:
        console.print(Panel(sg.result.copy_string(), title="Next settings to enter", expand=False))
        if sg.result.headroom_db > 0:
            console.print(f"[yellow]Headroom:[/] lower the S3 volume by {sg.result.headroom_db:.1f} dB.")
        console.print("Enter them, then run [bold]sweeptune verify[/] again.")


# ---------------------------------------------------------------------- plot
def frequency_plot(s: Session, channels: bool = False, out: Optional[Path] = None) -> Path:
    from . import report as rp

    cfg = state.cfg
    d = s.latest_measurement_dir()
    m = load_measurement(d)
    if not (d / "suggestion.json").exists():
        run_suggest(s, d, m, quiet=True)
    sugg = json.loads((d / "suggestion.json").read_text())
    mic = load_profile(m.mic)
    model = load_streamer_model(cfg)
    ra = analyze_measurement(m, mic, cfg, model)
    gains = np.asarray(sugg["gains"], dtype=float)
    eq_db = eq_curve(FREQS, gains, cfg.bands, model)
    eq_color = rp.SERIES[1]
    if channels:
        curves = [(f"{ch} speaker (measured)", ra.per_channel[ch]) for ch in ("L", "R") if ch in ra.per_channel]
        eq_color = rp.TEXT_2  # orange means "R" in this view
    else:
        state_eq = "with the EQ currently set" if any(m.applied_eq) else "EQ flat"
        curves = [(f"Measured ({state_eq})", ra.average),
                  ("Predicted with the EQ below", np.asarray(sugg["room"]) + eq_db)]
    title = "Frequency response at the listening position"
    if s.meta.get("simulate"):
        title += " (SIMULATED ROOM)"
    subtitle = (f"Session {s.path.name} · {m.kind} · mic {m.mic} · {len(m.positions())} positions"
                f"{' · L/R on a shared level (balance visible)' if channels else ', L+R'} · "
                f"1/6-octave smoothing · target {sugg['target']}")
    path = out or s.path / "frequency-response.png"
    return rp.plot_frequency_response(path, FREQS, curves, np.asarray(sugg["target_curve"]),
                                      np.asarray(sugg["target_adapted"]), cfg.bands, gains, m.applied_eq, eq_db,
                                      (mic.trust_low_hz, mic.trust_high_hz), title, subtitle, eq_color,
                                      shared_level=channels)


@app.command()
def plot(session: Optional[Path] = typer.Option(None, help="Session folder (default: latest)."),
         channels: bool = typer.Option(False, "--channels", help="Show L and R separately."),
         out: Optional[Path] = typer.Option(None, help="PNG path (default: <session>/frequency-response.png)."),
         show: bool = typer.Option(True, "--open/--no-open", help="Open the PNG (macOS Preview).")):
    """Plot the measured frequency response, the target and the suggested EQ."""
    try:
        s = Session.open(session)
        path = frequency_plot(s, channels, out)
    except FileNotFoundError as e:
        fail(str(e))
    console.print(f"Plot written to {path}")
    if show and sys.platform == "darwin" and sys.stdout.isatty():
        subprocess.run(["open", str(path)], check=False)


# ---------------------------------------------------------------------- report
@app.command()
def report(session: Optional[Path] = typer.Option(None)):
    """Write plots and a Markdown summary for the session."""
    from . import report as rp
    from .deconv import IR_PRE_S

    cfg = state.cfg
    try:
        s = Session.open(session)
        last_dir = s.latest_measurement_dir()
    except FileNotFoundError as e:
        fail(str(e))
    model = load_streamer_model(cfg)
    curves, history = [], []
    sugg = json.loads((last_dir / "suggestion.json").read_text()) if (last_dir / "suggestion.json").exists() else None
    if sugg is None:
        m = load_measurement(last_dir)
        run_suggest(s, last_dir, m, quiet=True)
        sugg = json.loads((last_dir / "suggestion.json").read_text())
    target = np.asarray(sugg["target_curve"])
    adapted = np.asarray(sugg["target_adapted"])
    m = mic = ra = None
    for it in s.meta["iterations"]:
        d = s.path / it["dir"]
        if it["kind"] not in ("baseline", "verify") or not (d / "measurement.json").exists():
            continue
        m = load_measurement(d)
        mic = load_profile(m.mic)
        ra = analyze_measurement(m, mic, cfg, model)
        label = f"{it['index']}: {it['kind']} ({format_eq(cfg.bands, m.applied_eq) if any(m.applied_eq) else 'flat'})"
        curves.append((label if len(label) < 70 else f"{it['index']}: {it['kind']}", ra.average))
        y = smooth(FREQS, ra.average, 1 / 3)
        y = y + level_offset(FREQS, y, adapted)
        err = band_levels(FREQS, y, cfg.bands) - band_levels(FREQS, adapted, cfg.bands)
        mask = [mic.trust_low_hz <= c <= mic.trust_high_hz and f"{c:g}" not in sugg.get("reasons", {})
                for c in cfg.bands]
        history.append({"index": it["index"], "kind": it["kind"], "applied": format_eq(cfg.bands, m.applied_eq),
                        "max_err": float(np.max(np.abs(err[mask])))})
    ac = acoustics_summary(m, ra)
    imgs = [
        frequency_plot(s),
        rp.plot_responses(s.path / "response.png", FREQS, curves, target, adapted, cfg.bands,
                          (mic.trust_low_hz, mic.trust_high_hz)),
        rp.plot_rt60(s.path / "rt60.png", ac["rt60"]),
        rp.plot_etc(s.path / "etc.png", m.irs[f"{m.positions()[0]}-{m.channels()[0]}"], m.fs, IR_PRE_S),
    ]
    md = rp.markdown(cfg, s.meta, sugg, ac, [p.name for p in imgs], history, bal.from_measurement(m.irs, m.fs))
    out = s.path / "report.md"
    out.write_text(md)
    console.print(f"Report written to {out}")
    for tip in rp.treatment_advice(ac):
        console.print(f"• {tip}")


if __name__ == "__main__":
    app()
