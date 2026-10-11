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

- **Launchers.** `Tag Library.bat` and `Sound Browser.bat` on Windows, `tag-library.sh` and `sound-browser.sh` on macOS and Linux: drag your sound folder on, or type it once. Everything below runs from them; nothing else needs the terminal.
- **`audio-designer` skill.** The method an agent follows: turn a request into a brief, shortlist from the library, recognise and measure the shortlist, return at most three candidates with their evidence (or "no suitable match"), then plan and check the mix: layer priority, starting levels, ducking, frequency clashes, a cue sheet, and render checks that work without listening.
- **`audio_tag.py`.** Tags any audio or video file with AudioSet sound-event labels (527 classes: birds, speech, choir, crickets, whoosh, piano…) for the whole file and for each 10-second window. It also ranks files against a written description with CLAP. It can tag an entire library into one CSV, resuming where it stopped.
- **`audio_map.py` and `audio_index.py`.** Measurements an agent can trust: tempo, beat times, loud and quiet sections and strongest hits of a track (for cutting video to music); loudness, true peak, frequency bands, event density, steadiness and a loop hint for ambience and effects. Each has a built-in self-test (`--selftest`, `check`).
- **`audio_search.py`: search by description, cross-referenced.** No single source is reliable on its own: names can be codes, the audio model confuses quiet files with "soft" sounds, and the AudioSet tagger mislabels short designed effects. So three sources rank the library separately: the **audio** fingerprint (CLAP), the **cleaned file name** ("FL_GFX_UI_Click_30.wav" becomes "ui click", matched by meaning and by exact words), and the **AudioSet labels**, which only confirm what the other two found. Reciprocal rank fusion merges them, so a file several sources agree on comes first, and every result keeps each source's own rank as evidence. The sound word of a query leads: in "a single soft wooden click", "click" counts most.
- **`tagging/vocab_tags.py`: tags that say who backs them.** Each file gets canonical studio tags (click, bird song, rain, choir, piano… 89 in all), each supported separately by the **name**, the **model labels** and the **audio**. Two or more sources make a tag; one makes a candidate. It also lists the files where sources contradict each other (the name says bird, nothing else hears one): the few clips worth a human listen.
- **Sound browser.** A local web page over the tagged library: filter by kind, pack, length and tag, search by description (with each source's rank shown on every result), find "more like this", audition with the keyboard, and like or reject sounds. Your verdicts are saved to a JSON file the agent reads before proposing anything.

## Get started

You need Python 3.10 or later and FFmpeg (with ffprobe) on your `PATH`. Then, from the repository folder:

```bash
pip install -r requirements.txt
```

With an NVIDIA GPU, install PyTorch's CUDA build first (`pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`): tagging tens of thousands of files is an overnight job on a GPU and many times longer on a CPU. Shortlists of a few hundred files are fine on a CPU.

**1. Tag your library.** On Windows, drag your sound folder onto **`Tag Library.bat`**; on macOS or Linux, run `./tag-library.sh "/path/to/sounds"`. It records what every file contains and a fingerprint for search, next to your sounds. Stop it any time: running it again continues where it stopped, and adds files you put in later. The folder is remembered in `library.txt`, so later runs need no argument.

**2. Browse it.** Double-click **`Sound Browser.bat`** (or run `./sound-browser.sh`). The page opens in your web browser; keep the window open while you use it. Files not tagged yet are still listed and playable, and **Reload** picks up a tagging pass still running.

On first use, the tools download two models into the Hugging Face cache (about 1 GB together; CED also needs `torchaudio`):

| Model | Purpose | Licence |
|---|---|---|
| [mispeech/ced-base](https://huggingface.co/mispeech/ced-base) | Sound-event labels (AudioSet), pinned to a reviewed revision because it runs code from its model page | Apache-2.0 |
| [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused) | Search by description, "more like this" | Apache-2.0 |

## The tools from the command line

Every tool lives in `skills/audio-designer/scripts/`, shared with the agent skill, and has a self-test:

```bash
python skills/audio-designer/scripts/audio_tag.py check
python skills/audio-designer/scripts/audio_search.py check
```

Tag a few loose files, or rank them against a description by their audio:

```bash
python skills/audio-designer/scripts/audio_tag.py tag bed.wav call.mp3
python skills/audio-designer/scripts/audio_tag.py rank "a short soft wooden knock" a.wav b.wav c.wav
```

Tag a whole library and search it from the terminal, with each source's evidence:

```bash
python skills/audio-designer/scripts/audio_tag.py tag --root "D:/MySounds" --out "D:/MySounds/AUDIO_TAGS.csv"
python skills/audio-designer/scripts/audio_search.py find --root "D:/MySounds" "a wooden click" --top 20
```

The library pass writes `AUDIO_TAGS.csv` (labels, 10-second timeline, speech/music/silence scores) and `AUDIO_TAGS_clap.f32` (one 512-number CLAP vector per file); the search adds `AUDIO_NAMES_clap.npz` (file-name vectors). All three stay next to your sounds.

## The sound browser

`Sound Browser.bat` / `sound-browser.sh` run this for you:

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

Tags show solid when two or more sources agree and faded with a dashed outline when only one source claims them (hover for which). The tag panel counts agreed tags unless you include candidates; the raw AudioSet labels sit in a collapsed section.

Each file shows a strip of 10-second blocks: blocks where a selected tag is heard light up, and clicking a block jumps there.

## How the tagger was chosen

The AudioSet tagger was CED-base's predecessor AST until a paired comparison on 513 clips sampled by file name across 15 sound families (`bench/tagger_trial.py`): CED-base named the file's family in its top five where AST did not on 59 clips, the reverse on 21 (exact McNemar p < 0.0001), at half the time per clip. It gained most where AST was weakest: clicks (0 to 15%), footsteps (10 to 42%), insects (10 to 25%). File names are a proxy and fail on some families ("Talking UI" sounds are not speech), so read it as a comparison, not an accuracy. `AUDIO_TAGGER=ast` keeps the old tagger available.

The search fingerprints changed too: short clips are padded with silence instead of looped to the model's 10-second window. A looped 0.4 s click reads as a rattle; on 633 click-named files plus 1,500 others, click hits in the top 100 rose from 38-60 to 69-77 across four click queries. `audio_tag.py reembed` applies it to an existing library in place.

## Measuring search quality

`skills/audio-designer/bench/` builds a stratified benchmark of a few hundred clips from your own tagged library (14 sound families, each with confusable hard negatives, whole packs kept on one side of a dev/test split) and scores any search method against it: P@10, P@50 and average precision per query, dev and test reported separately.

```bash
AUDIO_LIBRARY=D:/MySounds python skills/audio-designer/bench/bench_select.py build
AUDIO_LIBRARY=D:/MySounds python skills/audio-designer/bench/bench_eval.py run --mode proxy --callable skills/audio-designer/bench/methods.py:fused
```

Truth comes from you listening to each clip (about 20 seconds per clip, instructions in the bench README). Until then, `--mode proxy` scores against file names and says so on every report: it flattens differences and flatters name-based methods.

On the author's library, with the proxy: audio-only search scored a mean average precision of 0.26 (dev) and 0.25 (test); the fused search 0.37 and 0.35; the file name alone 0.56 and 0.53. The last number shows the proxy's bias rather than a better method, since names are also its answer key, so whether fusion beats names alone awaits the listening review. For "a single soft wooden click" over 22,000 files, click-named results in the top 100 went from 3 (audio only) to 60 (fused).

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
