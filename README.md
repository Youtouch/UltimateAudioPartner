# Ultimate Audio Partner

Tools and AI-agent skills for finding, identifying and mixing sound from a large local library, built for agents that cannot hear and for the person who can.

An agent choosing sound by filename picks a "jungle ambience" with no animals in it, or calm music with a choir under it. These tools give it two better kinds of evidence and keep them separate from the third, which only a person has:

| Evidence | Where it comes from | What it can prove |
|---|---|---|
| **Measured** | FFmpeg and NumPy meters: loudness, peaks, frequency bands, event density, beats | Length, level, shape. Not what the sound is. |
| **Recognised** | `audio_tag.py`: sound-event labels per 10 seconds, plus text-to-audio similarity | What a model hears in the file, with a probability |
| **Judged** | Your ear, saved from the browser as likes, rejects and notes | Whether it fits. Only this settles taste. |

Everything runs locally. No account, no server, nothing running in the background.

## What is in it

- **`audio-designer` skill.** The method an agent follows: turn a request into a brief, shortlist from the library, recognise and measure the shortlist, return at most three candidates with their evidence (or "no suitable match"), then plan and check the mix: layer priority, starting levels, ducking, frequency clashes, a cue sheet, and render checks that work without listening.
- **`audio_tag.py`.** Tags any audio or video file with AudioSet sound-event labels (527 classes: birds, speech, choir, crickets, whoosh, piano…) for the whole file and for each 10-second window. It also ranks files against a written description with CLAP. It can tag an entire library into one CSV, resuming where it stopped.
- **`audio_map.py` and `audio_index.py`.** Measurements an agent can trust: tempo, beat times, loud and quiet sections and strongest hits of a track (for cutting video to music); loudness, true peak, frequency bands, event density, steadiness and a loop hint for ambience and effects. Each has a built-in self-test (`--selftest`, `check`).
- **Sound browser.** A local web page over the tagged library: filter by kind, pack, length and tag, search by description, find "more like this", audition with the keyboard, and like or reject sounds. Your verdicts are saved to a JSON file the agent reads before proposing anything.

## Requirements

- Python 3.10 or later
- FFmpeg and ffprobe on `PATH` (or set `FFMPEG` to the `ffmpeg` executable)
- Python packages:

```bash
pip install numpy scipy soundfile librosa transformers
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

The second line installs PyTorch with CUDA. Without an NVIDIA GPU, `pip install torch` gives the CPU build: fine for shortlists of a few hundred files, but tagging a library of tens of thousands of files takes many hours on a CPU.

On first use, two models download into the Hugging Face cache (about 1 GB together):

| Model | Purpose | Licence |
|---|---|---|
| [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | Sound-event labels (AudioSet) | BSD-3-Clause |
| [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused) | Search by description, "more like this" | Apache-2.0 |

## Quick start

Check the installation (it tags a test tone and test noise, and checks CLAP ranks them correctly):

```bash
python skills/audio-designer/scripts/audio_tag.py check
```

Tag a few files:

```bash
python skills/audio-designer/scripts/audio_tag.py tag bed.wav call.mp3
```

Rank files against a description:

```bash
python skills/audio-designer/scripts/audio_tag.py rank "a short soft wooden knock" a.wav b.wav c.wav
```

Tag a whole library (resumable: run the same command again to continue, or to add new files):

```bash
python skills/audio-designer/scripts/audio_tag.py tag --root "D:/MySounds" --out "D:/MySounds/AUDIO_TAGS.csv"
```

This writes `AUDIO_TAGS.csv` (labels, 10-second timeline, speech/music/silence scores) and `AUDIO_TAGS_clap.f32` (one 512-number CLAP vector per file) next to your sounds.

## The sound browser

```bash
python skills/audio-designer/scripts/sound_browser.py --root "D:/MySounds"
```

It opens `http://127.0.0.1:8790/` in your browser, listening on your own machine only, and stops when you close the window or press Ctrl+C. If the port is busy, it takes the next free one.

It reads, from the library root:

- `AUDIO_TAGS.csv` and `AUDIO_TAGS_clap.f32` from the tagging pass. Untagged files are listed too, marked "Not tagged yet"; **Reload** picks up a pass still running.
- `AUDIO_LEDGER_files.csv` if present (columns `relative_path`, `pack`, `duration_seconds`, `bucket`), for pack names, lengths and kinds. Without it, the pack is the top folder.

It writes `AUDIO_VERDICTS.json` there: your likes, rejects and notes.

| Key | Action |
|---|---|
| ↓ / ↑ | Next / previous file, and play it |
| Space | Play / pause |
| ← / → | Back / forward 5 seconds |
| L / X | Like / reject |
| M | More like this |
| C | Copy the full path |

Each file shows a strip of 10-second blocks: blocks where a selected tag is heard light up, and clicking a block jumps there.

## Using the skills with an agent

Copy the folders under `skills/` into your agent's skill directory: `~/.claude/skills/` for Claude Code, `~/.agents/skills/` for Codex. The agent then loads `audio-designer` when it picks, identifies or mixes sound. A project can add a short sound profile listing approved and rejected sounds; the skill reads it first.

## What the tools cannot do

- **They do not hear.** A label is a model's reading, trained on YouTube audio. It is strongest on animals, weather, water, speech, crowds and instruments, and weaker on designed effects and synthetic interface sounds.
- **Short clips score low.** A two-second call is padded into a ten-second window, so a modest label on a short file can still be a real hit.
- **Description search is loose.** Words like "soft" also match near-silent files. Narrow by kind and length first.
- **Measurements and labels do not judge fit.** Whether a sound works in your mix is still a listen.
- **Rights are yours to check.** Tagging a file says nothing about permission to use it.

## Licence

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The models are downloaded under their own licences, listed above. No sounds or library data are included or should be committed: the `.gitignore` excludes them.
