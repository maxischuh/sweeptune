"""PNG plots and a Markdown summary for a session."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import level_offset, smooth  # noqa: E402
from .config import Config, band_label  # noqa: E402
from .eq import format_eq  # noqa: E402

RT60_TARGET = (0.3, 0.5)

# Chart tokens (validated categorical slots 1-2 on the light surface)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
NEUTRAL_BAND = "#f0efec"
SERIES = ("#2a78d6", "#eb6834")


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.grid(True, which="minor", axis="x", color=GRID, lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)


def plot_frequency_response(path: Path, freqs, curves: list[tuple[str, np.ndarray]], target, target_adapted,
                            bands, gains, current, eq_db, trust: tuple[float, float], title: str,
                            subtitle: str, eq_color: str = SERIES[1], shared_level: bool = False) -> Path:
    """Two panels sharing the frequency axis: in-room response vs target, and the S3 EQ settings.

    Each curve is level-aligned to the target at 500 Hz-2 kHz; with `shared_level`
    all curves get one common offset so differences between them (L/R balance) stay visible.
    """
    fig, (ax, ax_eq) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, facecolor=SURFACE,
                                    gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.12})
    for a in (ax, ax_eq):
        _style_axes(a)
        lo, hi = trust
        if lo > 20:
            a.axvspan(20, lo, color=NEUTRAL_BAND, zorder=0)
        if hi < 20000:
            a.axvspan(hi, 20000, color=NEUTRAL_BAND, zorder=0)

    ax.semilogx(freqs, target, color=TEXT_2, lw=1.2, ls="--", label="Target", zorder=2)
    if not np.allclose(target, target_adapted):
        ax.semilogx(freqs, target_adapted, color=TEXT_2, lw=1.0, ls=":",
                    label="Target, following speaker roll-off", zorder=2)
    in_view = (freqs >= 20) & (freqs <= 20000)
    lo, hi = float(np.min(target[in_view])), float(np.max(target[in_view]))
    common = None
    if shared_level:
        from .analysis import power_average

        mean = smooth(freqs, power_average([db for _, db in curves]), 1 / 6)
        common = level_offset(freqs, mean, target_adapted)
    for (label, db), color in zip(curves, SERIES):
        y = smooth(freqs, db, 1 / 6)
        y = y + (common if common is not None else level_offset(freqs, y, target_adapted))
        ax.semilogx(freqs, y, color=color, lw=2.0, label=label, zorder=3)
        lo, hi = min(lo, float(np.min(y[in_view]))), max(hi, float(np.max(y[in_view])))
    ax.set_ylim(min(-20.0, np.floor(lo / 5) * 5 - 5), max(18.0, np.ceil(hi / 5) * 5 + 3))
    ax.set_ylabel("Level (dB, 0 dB = target at 1 kHz)", color=TEXT_2, fontsize=9)
    ax.legend(loc="lower left", fontsize=9, frameon=False, labelcolor=TEXT)
    fig.suptitle(title, x=0.075, ha="left", y=0.975, fontsize=14, color=TEXT, fontweight="bold")
    fig.text(0.075, 0.925, subtitle, ha="left", fontsize=9.5, color=TEXT_2)
    y_top = ax.get_ylim()[1] - 1.5
    if trust[1] < 20000:
        ax.text(np.sqrt(trust[1] * 20000), y_top, "mic not\ntrusted", ha="center", va="top", fontsize=8, color=MUTED)
    if trust[0] > 20:
        ax.text(np.sqrt(trust[0] * 20), y_top, "mic not\ntrusted", ha="center", va="top", fontsize=8, color=MUTED)

    # EQ panel: one bar per S3 band (what to enter), plus the resulting filter curve
    gains = np.asarray(gains, dtype=float)
    lefts = np.asarray(bands) * 2 ** -0.22
    widths = np.asarray(bands) * (2 ** 0.22 - 2 ** -0.22)
    ax_eq.bar(lefts, gains, width=widths, align="edge", color=eq_color, alpha=0.35, zorder=2)
    ax_eq.semilogx(freqs, eq_db, color=eq_color, lw=2.0, zorder=3)
    ax_eq.axhline(0, color=MUTED, lw=0.8, zorder=1)
    cur = np.asarray(current, dtype=float)
    if np.any(cur):
        ax_eq.scatter(bands, cur, marker="_", s=260, color=TEXT, lw=2, zorder=4, label="currently set")
        ax_eq.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=TEXT)
    lim = max(4.0, float(np.max(np.abs(np.concatenate([gains, eq_db])))) + 2.5)
    ax_eq.set_ylim(-lim, lim)
    for b, g in zip(bands, gains):
        ax_eq.text(b, g + (0.6 if g >= 0 else -0.6), f"{g:+.1f}", ha="center",
                   va="bottom" if g >= 0 else "top", fontsize=8.5, color=TEXT)
    ax_eq.set_ylabel("S3 EQ (dB)", color=TEXT_2, fontsize=9)
    ax_eq.set_xlabel("Frequency (Hz)", color=TEXT_2, fontsize=9)
    ax_eq.set_xlim(20, 20000)
    ax_eq.set_xticks(list(bands), [band_label(b) for b in bands])
    ax_eq.xaxis.set_minor_formatter(plt.NullFormatter())
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def treatment_advice(ac: dict) -> list[str]:
    tips = []
    rt = ac["rt60"]
    mid = ac["rt60_mid"]
    if np.isfinite(mid):
        if mid > RT60_TARGET[1] + 0.1:
            tips.append(f"Reverberation is high (RT60 {mid:.2f} s at 500 Hz-2 kHz, aim for "
                        f"{RT60_TARGET[0]}-{RT60_TARGET[1]} s): add soft furnishings - a thick rug between "
                        "speakers and seat, curtains, a bookshelf or absorber panels on the back wall.")
        elif mid < 0.25:
            tips.append(f"The room is very dead (RT60 {mid:.2f} s): avoid adding more absorption; "
                        "diffusion (bookshelves) keeps it lively.")
        else:
            tips.append(f"Mid/high reverberation is in a good range (RT60 {mid:.2f} s).")
    low = [rt.get(b, np.nan) for b in (63, 125)]
    low = [v for v in low if np.isfinite(v)]
    if low and np.isfinite(mid) and np.mean(low) > 1.5 * mid and np.mean(low) > 0.5:
        tips.append(f"Bass decays much slower ({np.mean(low):.2f} s at 63-125 Hz) than the mids: "
                    "bass traps in the front corners help; EQ only removes the level peak, not the ringing.")
    for ch, refl in ac["reflections"].items():
        strong = [r for r in refl if r["level_db"] > -10]
        for r in strong[:2]:
            tips.append(f"Strong early reflection on {ch} at {r['delay_ms']} ms ({r['level_db']} dB, "
                        f"~{r['extra_path_cm']:.0f} cm extra path): find the first-reflection point "
                        "(mirror trick on side wall / desk / floor) and absorb it.")
    for p in ac["modes"]["peaks"]:
        if p["prominence_db"] >= 4:
            tips.append(f"Room mode at {p['freq']} Hz (+{p['prominence_db']} dB, Q~{p['q']}): the EQ cut "
                        "helps; moving the speakers or seat by 20-50 cm can help more.")
    for n in ac["modes"]["nulls"]:
        tips.append(f"Null at {n['freq']} Hz (-{n['depth_db']} dB): EQ can't fill a cancellation. "
                    "Move the seat or speakers (or add the subwoofer later) instead.")
    tips.append("EQ cannot remove echo: flutter or slap echo needs absorption or diffusion on "
                "parallel bare walls.")
    return tips


def plot_responses(path: Path, freqs, curves: list[tuple[str, np.ndarray]], target, target_adapted,
                   bands, trust: tuple[float, float]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, db in curves:
        y = smooth(freqs, db, 1 / 6)
        y = y + level_offset(freqs, y, target_adapted)
        ax.semilogx(freqs, y, lw=1.6, label=label)
    ax.semilogx(freqs, target, "k--", lw=1.2, label="target")
    if not np.allclose(target, target_adapted):
        ax.semilogx(freqs, target_adapted, "k:", lw=1.2, label="target (speaker roll-off)")
    lo, hi = trust
    if lo > freqs[0]:
        ax.axvspan(freqs[0], lo, color="0.85", label="low mic confidence")
    if hi < freqs[-1]:
        ax.axvspan(hi, freqs[-1], color="0.85")
    for b in bands:
        ax.axvline(b, color="0.7", lw=0.5)
    ax.set_xticks(list(bands), [band_label(b) for b in bands])
    ax.set_xlim(20, 20000)
    ax.set_ylim(-25, 20)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("dB (aligned to target at 500 Hz-2 kHz)")
    ax.set_title("In-room response (1/6-octave, multi-position average)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_rt60(path: Path, rt: dict) -> Path:
    bands = list(rt.keys())
    vals = [rt[b] for b in bands]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar([band_label(b) for b in bands], [v if np.isfinite(v) else 0 for v in vals], color="#4c78a8")
    ax.axhspan(*RT60_TARGET, color="#59a14f", alpha=0.2, label="typical living-room target")
    ax.set_ylabel("RT60 (s)")
    ax.set_xlabel("Octave band (Hz)")
    ax.set_title("Reverberation time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_etc(path: Path, ir: np.ndarray, fs: int, pre_s: float) -> Path:
    from scipy import signal

    env = np.abs(signal.hilbert(ir))
    t0 = int(pre_s * fs)
    d = t0 + int(np.argmax(env[t0 - int(0.001 * fs): t0 + int(0.002 * fs)])) - int(0.001 * fs)
    db = 20 * np.log10(np.maximum(env / env[d], 1e-6))
    t = (np.arange(len(ir)) - d) / fs * 1000
    m = (t >= -2) & (t <= 60)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t[m], db[m], lw=0.8)
    ax.set_ylim(-50, 3)
    ax.set_xlabel("Time after direct sound (ms)")
    ax.set_ylabel("dB")
    ax.set_title("Energy-time curve (first position, left)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def markdown(cfg: Config, session_meta: dict, sugg: dict | None, ac: dict, images: list[str],
             history: list[dict], balance=None) -> str:
    lines = ["# Room calibration report", "",
             f"- Session: `{session_meta.get('created', '')}`",
             f"- Mic: `{session_meta.get('mic', '')}` · Output: `{session_meta.get('output', '')}`",
             f"- Target: `{session_meta.get('target', cfg.target)}`", ""]
    if sugg:
        lines += ["## EQ settings for the Fosi S3", "",
                  "| Band | Setting | Expected residual | Note |", "|---|---|---|---|"]
        for c, g, r in zip(cfg.bands, sugg["gains"], sugg["residual_bands"]):
            note = sugg.get("reasons", {}).get(f"{c:g}", "")
            lines.append(f"| {band_label(c)} Hz | {g:+.1f} dB | {(f'{r:+.1f} dB' if r is not None and np.isfinite(r) else 'n/a')} | {note} |")
        lines += ["", f"Copy string: `{format_eq(cfg.bands, sugg['gains'])}`", ""]
        if sugg["headroom_db"] > 0:
            lines += [f"**Headroom:** the EQ boosts up to {sugg['headroom_db']:.1f} dB, so lower the "
                      "S3 volume by that much to avoid digital clipping.", ""]
    if balance is not None:
        lines += ["## Channel balance (amp)", "",
                  f"R − L at 500 Hz–4 kHz: **{balance.r_minus_l_db:+.1f} dB** "
                  f"(spread over {balance.positions} positions ±{balance.spread_db:.1f} dB)", "",
                  balance.advice(cfg.eq_step_db), "",
                  "| Octave | R − L |", "|---|---|"]
        lines += [f"| {band_label(b)} Hz | {v:+.1f} dB |" for b, v in balance.per_octave.items()]
        lines += ["", "Differences outside 500 Hz–4 kHz come from the room and speaker placement, "
                  "not from the balance control.", ""]
    if history:
        lines += ["## Iterations", "", "| # | Kind | Applied EQ | Largest band error vs target |", "|---|---|---|---|"]
        for h in history:
            lines.append(f"| {h['index']} | {h['kind']} | `{h['applied']}` | {h['max_err']:.1f} dB |")
        lines.append("")
    lines += ["## Room acoustics", "", "| Octave | RT60 (s) |", "|---|---|"]
    for b, v in ac["rt60"].items():
        lines.append(f"| {band_label(b)} Hz | {v:.2f} |" if np.isfinite(v) else f"| {band_label(b)} Hz | n/a |")
    lines += ["", "### Early reflections (first 20 ms)", ""]
    for ch, refl in ac["reflections"].items():
        if refl:
            items = ", ".join(f"{r['delay_ms']} ms / {r['level_db']} dB / +{r['extra_path_cm']:.0f} cm" for r in refl)
            lines.append(f"- **{ch}:** {items}")
        else:
            lines.append(f"- **{ch}:** none above -20 dB")
    lines += ["", "## What to do about the room", ""] + [f"- {t}" for t in treatment_advice(ac)]
    lines += [""] + [f"![{Path(i).stem}]({i})" for i in images]
    return "\n".join(lines) + "\n"
