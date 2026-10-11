# Read a track's beats and energy

Use when a cut has to follow music, or when a reference video's sound has to be described: where the beats fall, which passages are loud or quiet, where the track rises or drops. An agent cannot hear. [audio_map.py](../scripts/audio_map.py) measures instead, and the numbers are enough to place cuts.

Status: **ran on 2026-10-07** with librosa 1.0.0, Python 3.12 and FFmpeg 9.0.2 on Windows. The self-test (120 BPM clicks, quiet then loud) passed. A synthetic 30-second MP4 with a 128 BPM pulse, a quiet first 8 seconds and a quieter last 8 was read through FFmpeg: 65 beats covering the whole file, section boundaries at 8.01 s and 21.59 s against true changes at 8 s and 22 s. **No real music track has been analysed yet**, so the section and hit output on dense material is unproven. Treat the first real run as the test and check it against the user's ear.

Dependencies: Python with `librosa` (`pip install librosa`), and `ffmpeg` on PATH to decode files. If librosa is missing, name the dependency and ask; do not install it unasked. The rest of the pack works without it.

For ambience and animal calls, use [sound-without-hearing.md](sound-without-hearing.md) and `scripts/audio_index.py`. Beat estimates do not establish audible wildlife, a coherent biome or mix balance.

## Run it

```text
python scripts/audio_map.py track.mp3
python scripts/audio_map.py capture.mp4 --sections 5 --hits 20
python scripts/audio_map.py --selftest
```

Any audio or video file FFmpeg can decode. Output is one JSON object; all times are seconds from the start of **that file**, so after a trim, offset them or analyse the trimmed audio.

| Field | Meaning | Use |
|---|---|---|
| `tempo_bpm` | Average tempo over the tracked beats | Shot length in beats: at 120 BPM one beat is 0.5 s, a four-beat bar is 2 s |
| `beats` | Time of every beat | Candidate cut points |
| `sections` | Passages that sound alike, with `level_db` (0 is the loudest section, more negative is quieter) | The track's shape: intro, build, peak, breakdown, outro |
| `strongest_hits` | The sharpest transients | Anchor an impact, a reveal or a logo to one |
| `energy_db_per_second` | Loudness of each second, 0 is the loudest | Find a rise or drop inside a section |

## Place the cut

1. Edit for meaning first, as video-edit-director's [finish-and-rescue.md](../../video-edit-director/references/finish-and-rescue.md) says. The beat grid chooses between two frames that both work; it does not decide what the shot is.
2. Read the shape from `sections`. Put the hook on the first passage that carries energy, the payoff on the loudest, and the end card on the outro or a final hit. If the loudest passage starts too late for the video's length, start the music later in the track instead of stretching the picture.
3. Move a cut to the nearest beat only when that costs a few frames. If the nearest beat would clip an input, a reaction or a caption's reading time, keep the meaning and miss the beat.
4. Land a reveal on a boundary where `level_db` jumps up, or on a strong hit. A boundary where it falls suits a pause, a held result or the call to action.
5. Convert times to frames at the output frame rate and write them into the edit data. Do not round by eye.

Count shot lengths in whole beats when the edit is a montage or trailer. Do not force a pulse onto speech, slow deduction or ambient music.

## First real track, 2026-10-08

A 28-second hip hop cut under a game advert. What the run added:

- **A constant beat grid drifts.** A hand-made estimate (0.418 s from 0.01 s) against the tracked beats (mean 0.413 s from 0.14 s, each gap between 0.35 and 0.49 s) was a quarter of a second out by the end. Use the `beats` list itself for cut times, not `tempo_bpm` multiplied out.
- **The tempo came back doubled** (145 against a felt 72). The strong hits, four beats apart, were the bar: put payoffs on those and smaller events on the beats between.
- **Build the edit round the track's lift.** The first draft put the loudest passage under the end card. Moving the busiest picture (a montage, one still per beat, then two per beat) onto the passage where `level_db` jumps, and landing the logo on a strong hit after it, was the single change the user reacted to most.
- **If the agent controls the capture, capture in tempo.** Scripted inputs one or two beats apart mean the game's own sounds fall in time, and one anchor event per shot is enough to align the rest: choose the source offset so that event sits on its strong hit.
- **Still listen-checked by a person.** The user confirmed the cuts felt on the music; the agent could only confirm the numbers.

## What the numbers cannot tell you

- **Whether it sounds good.** Mix quality, mood and whether a moment lands are still a human listen. Say so in the report.
- **Tempo can be double or half** the felt tempo. Check that `beats` spacing matches `tempo_bpm`, then ask which reading matches the feel when it matters.
- **Speech, ambience and short sound effects have no beat.** The tool still prints a tempo and beats for them; ignore both and use only the energy fields.
- **Beats can be missing** in a passage of true digital silence, and a section a fraction of a second long at either end of the file is an edge artifact.
- **Section count is a guess** from the duration. Pass `--sections` when the structure is known, and merge neighbours with similar levels when reading.
- **Rights.** Measuring a track says nothing about permission to use it.

**Proof check:** the self-test passes; beat spacing agrees with the tempo; on the rendered export, extract a frame at two or three planned cut times and confirm each cut is there; the user or another listener confirms the cuts feel on the music.
