# sweeptune

Minimal macOS CLI that measures your listening room with a log sine sweep and tells you what to
enter into the **Fosi Audio S3** 10-band EQ (31.5 Hz – 16 kHz, ±10 dB, 0.5 dB steps) to reach a
neutral-but-pleasing, Harman-like in-room curve. It also reports reverberation (RT60), early
reflections and room modes, because EQ cannot fix echo.

## Setup

```bash
brew install uv
uv sync
uv run sweeptune --help
```

Select the Fosi S3 as the Mac's sound output (Control Center → Sound → AirPlay) before measuring.
Device names can be overridden in an optional `sweeptune.toml` (see `src/sweeptune/config.py` for all keys):

```toml
output_device = "S3"        # name fragment of the AirPlay output
input_device = ""           # name fragment; empty = mic profile default
max_boost_db = 4.0
shelf_db = 6.0              # bass shelf of the harman-like target
```

## Workflow

```bash
uv run sweeptune doctor                         # routing, mic inputs, noise floor
uv run sweeptune balance --live                 # set the amp's balance knob at the seat (L/R level + distance)
uv run sweeptune measure --mic audyssey         # 7 positions, L and R, EQ flat
uv run sweeptune eq suggest                     # table + copy string for the S3
# enter the values on the S3 (and lower its volume by the headroom shown)
uv run sweeptune verify                         # re-measure, compare to prediction, next correction
uv run sweeptune verify                         # repeat until converged
uv run sweeptune report                         # sessions/<timestamp>/report.md + PNGs
```

### Balance

`sweeptune balance --live` plays L, R, L sweeps at the listening seat and shows how much louder one side is
(500 Hz – 4 kHz, where the stereo image lives) and how much farther one speaker is. Turn the amp's balance
knob, press Enter to re-measure, and stop within ±0.5 dB. Fix a distance difference by moving the
speaker first; the balance control is for the remaining level difference. Without `--live`,
`sweeptune balance` reports the same figures from the latest multi-position measurement. The S3 EQ acts on both
channels equally, so balance and EQ can be set in either order.

Optional:

- `sweeptune eq probe 125` – change one band by −6 dB and learn the S3's real filter Q (saved to `streamer.json`).
- `sweeptune mic relcal iphone17pro --ref audyssey` – derive a calibration for another mic at the same spot.
- `sweeptune sweep export` + `sweeptune analyze rec1.wav rec2.wav …` – offline path, e.g. recording on the iPhone
  with an app that offers a measurement (unprocessed) mode.
- `sweeptune --simulate …` – run any command against a synthetic room, no speakers needed.

## Microphones

| id | Mic | Use |
|---|---|---|
| `audyssey` | Audyssey mic bundled with AV receivers, MacBook 3.5 mm jack | primary; trusted 20 Hz – 10 kHz |
| `iphone17pro` | iPhone 17 Pro via Continuity mic, or offline WAV | secondary; AGC check first |
| `macbook-m1pro` | built-in array | last resort; 100 Hz – 8 kHz, compare iterations only |

None of these mics has a traceable calibration, so all results are relative. The MacBook jack is CTIA TRRS;
if `doctor` finds no external mic, the Audyssey plug needs a TRS→TRRS mic adapter.

## Tests

```bash
uv run pytest
```

## License

[MIT](LICENSE) © 2026 Maximilian Schuh. Free to use, modify and redistribute, including commercially, as long
as the copyright notice and license text are kept — i.e. with acknowledgement.
