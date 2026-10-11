# Choosing sound without hearing it

An agent cannot hear. Choose from names plus measurements, say so in the hand-off,
and let the user's ear decide. A filename is a guess, not evidence that animals are audible.

## Measure a shortlist

Run from this skill folder. Requires Python 3.10+, numpy and soundfile. No new packages.
FFmpeg/ffprobe use `FFMPEG` (an executable path), the WinGet Gyan bundle, or PATH.

```text
python scripts/audio_index.py index <library>
python scripts/audio_index.py index <library> --only "*forest*" --out forest.csv
python scripts/audio_index.py show path/to/bed.wav
python scripts/audio_index.py check
```

Default selection uses `AUDIO_LEDGER_files.csv`: `relative_path` and `kind_guess`
marked `sfx` or `voice`. Music and unknown entries are excluded. `--only` bypasses
that classification with a relative-path glob. The CSV and its companion Markdown
use forward-slash paths relative to the root, as the ledger does. Read its licence
column before use. A resumed run skips unchanged size/mtime rows and saves each new row.

| Measurement | Read it this way |
|---|---|
| Duration/channels | Whole file, including silence and tails |
| LUFS, LRA, true peak | FFmpeg `ebur128=peak=true`, first 60 s or the shorter whole file |
| `meter_input` | `file` normally; `decoded_pcm` if soundfile rescued a container FFmpeg rejected |
| Five energy shares | 0-300, 300-1500, 1500-3500, 3500-8000, 8000+ Hz; channel powers summed |
| Event density | Distinct 20 ms RMS rises above the file's own floor, per minute |
| Steadiness | Coefficient of variation of 1 s linear RMS; lower is steadier |
| Loop hint | Actual first/last 0.5 s within 3 dB and band-share distance <= 0.2 |
| Biome/source/role guesses | Folder/file names; short active calls versus longer beds |

Density ignores slow drift and has a 180 ms separation between onsets. Its threshold
is the larger of 2.5 times the 20th-percentile floor and median plus 10 median absolute
deviations. It separates chirps from steady wash; voice, footsteps and impacts can also
score. It does not identify animals. A loop hint is not proof of a clean seam.
Decoded-PCM fallback lengths use decodable samples when a short damaged header overstates duration.
Energy, density and CV use the same first 60 s; inspect longer files' other passages separately.

## Assemble one place

- Use one biome from one source in a scene. Never mix biomes in one scene.
- A call belongs to its bed. Use the bed's own pack where possible; the index names pairs.
- When asked for "nature sounds", animals must be heard. Prefer measured event density
  over steady atmosphere, then ask for listening approval in the hand-off.
- Put a bed about 30 dB under the voice before final loudness normalisation. In the
  worked mix, the bed was about -47 LUFS against a -16 LUFS voice mix. Meter the
  actual stem after gain; these are starting levels, not universal mastering targets.
- Bright beds, with most energy above 3.5 kHz (insects/crickets), read louder than
  their LUFS. Go lower still. Meter checks do not prove speech is clear.
- One call as punctuation after a joke starts about 10-14 dB under the voice.
  Short calls may not yield useful integrated LUFS; also check their peak and placement.
- Vary the biome between episodes and record the files, pack and biome in the series ledger.

## What the hand-off says

Report the file, source pack, LUFS/LRA/true peak, bright-band share, event density,
steadiness, loop hint, gain and cue time. Explain why it fits. Include **listening pending**.
Name guesses as guesses. A person confirms audible animals, coherent setting and balance.

In one narrated series, a jungle bed chosen by name was rejected because it had no audible
animals. In the next episode, a loud night-insect chorus read as savanna and an owl from another pack
felt misplaced. The accepted direction was one biome/source, audible animals, a bed
about -47 LUFS before the -16 LUFS mix's final normalisation, and one animal call after
a joke. The general rule travels; the episode's exact files stay in its project profile.
