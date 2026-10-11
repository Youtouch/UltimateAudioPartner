---
name: audio-designer
description: "Use when choosing, identifying or mixing sound for a video or game: what an audio file actually contains, finding music, ambience, animal calls, UI or effect sounds in a library, building a cue sheet, setting levels and ducking between voice, music, ambience and game sound, or checking a mix. Also when a sound 'is wrong', 'too loud', 'not heard' or 'doesn't fit'."
---

# Audio Designer

You cannot hear. Every claim about a sound rests on one of three kinds of evidence, and the hand-off names which:

| Evidence | Source | Proves |
|---|---|---|
| **Measured** | `audio_index.py`, `audio_map.py`, FFmpeg meters | Length, loudness, peaks, bands, density, beats. Not what the sound is. |
| **Recognised** | `scripts/audio_tag.py` (AudioSet labels per 10 s, CLAP ranking) | What a model hears in it, with a probability. A reading, not a listen. |
| **Judged** | The user's ear, recorded in the project profile | Whether it fits. Only this settles taste. |

A filename or folder is a fourth, weaker kind: a guess. Never write "contains birds" from a name.

Read the project's sound profile first if it has one (`.claude/audio-designer.md` in the project): approved and rejected sounds, the voice, starting levels, the library ledger. Past verdicts beat any rule here.

## Choose a sound

Read [recognise-and-select.md](references/recognise-and-select.md). In short:

1. Turn the request into a brief: role, character, prominence, how often it plays, timing, what plays with it, hard constraints.
2. Shortlist from the library's ledger and indexes by name (cheap), then **recognise** the shortlist with `audio_tag.py tag` and **measure** it with `audio_index.py show`. Drop a file when a label positively contradicts the brief (a calm track tagged `Choir` or `Chant`); when an expected label is merely missing (a "jungle" bed with no bird or frog label), keep it as doubtful and say so. A missing label is not an absence.
3. For a descriptive brief ("soft playful click") over a tagged library, run `audio_search.py find`: it ranks by three sources separately (the audio fingerprint, the cleaned file name, the AudioSet labels as confirmation only) and merges them, so a file several sources agree on comes first. Read each result's per-source ranks: a file found by the name alone or the audio alone is weaker evidence than one found by both. For loose files, `audio_tag.py rank` compares audio only. Either way a rank orders candidates; it is not taste.
4. Return at most three candidates as audiocards, with the exact excerpt (start/end), or say **no suitable match**. Never pad a shortlist.

For ambience and animal calls also read [sound-without-hearing.md](references/sound-without-hearing.md); for cutting to music, [music-and-beats.md](references/music-and-beats.md).

## Mix

Read [mix-plan.md](references/mix-plan.md) before rendering anything with more than one layer. Write the cue sheet (layer, file, excerpt, cue time, gain, ducking) first, render, then meter **each stem after gain** and **each stretch of the export**. Levels set by an agent are starting points: name the ones the user must judge.

## Hand off

Per sound: file, excerpt, role, the three kinds of evidence (labels with probabilities, key measurements, any past verdict), gain and cue time, and **listening pending**. After the user listens, record the verdict and its reason in the project profile, so the next choice starts from it.

## The user's browser

`python scripts/sound_browser.py --root <library>` opens a local page where the user browses the library by tag, kind, pack and length, searches by description, finds "more like this", and auditions with the arrow keys. Their likes, rejects and notes land in `AUDIO_VERDICTS.json` in the library root: **read it as judged evidence** before proposing a sound, and never re-propose a rejected file. To hand the user a shortlist, give paths they can paste into the name filter.

## Tools

Run from this skill folder. `audio_map.py` holds the shared loader and FFmpeg lookup (`FFMPEG`, the WinGet Gyan bundle, or PATH); other skills (such as a video export checker) may import it from here, so keep this folder's name.

```text
python scripts/audio_tag.py tag clip.wav other.mp3
python scripts/audio_tag.py tag --root <library> --only "*animals*" --out tags.csv
python scripts/audio_tag.py rank "gentle wooden click" a.wav b.wav c.wav
python scripts/audio_search.py find --root <library> "a wooden click" --top 20
python scripts/audio_tag.py check
```

**A tagged library beats a fresh run.** If the library has a tags CSV (the project profile says where), grep it first: `top_labels` and `timeline` per file, `speech`/`music`/`silence` scores, and `clap_row`, the row of that file's CLAP vector in `<csv>_clap.f32` (512 float32 per row, unit length; a dot product with `embed_text` or another row is the similarity). A changed file appends a row: keep the last row per path. Rerunning the same command tags only new or changed files.

**Searching by words.** The sound word of a query leads: in "a single soft wooden click" it is "click", and "soft" or "wooden" only refine. Describing words alone pull toward the wrong files ("soft" finds near-silence), so write the sound first. `audio_search.py names` caches each cleaned file name's text vector in `AUDIO_NAMES_clap.npz` beside the tags; it updates itself for new names.

**Studio tags with agreement.** `tagging/vocab_tags.py build --root <library>` writes `AUDIO_VOCAB_TAGS.csv`: per file and canonical tag, which sources support it (name, model label, audio). Two or more sources make a tag worth stating; one source is a candidate to verify, never a fact. `disagreements` lists the files where sources contradict each other: offer those for a listen before anything else. Rebuild it after a tagging or relabel pass.

**Judging a change to search or tagging.** `bench/` holds a reviewed-clip benchmark and `bench_eval.py`: run a method against it before and after, on the dev split while tuning and on test once. Until the clips are reviewed by ear it scores against file names (PROXY), which flatters name-based methods; say so with any number.

Needs `torch`, `torchaudio` and `transformers`; the models (CED-base for AudioSet labels, LAION CLAP for search) download to the Hugging Face cache on first use. CED runs code from its model page, pinned to a reviewed revision; `AUDIO_TAGGER=ast` switches back to the older AST tagger. Every tags row records its `tagger`. `relabel` re-labels an existing tags file with the current tagger, keeping fingerprints; `reembed` silence-pads short files' fingerprints (a looped 0.4 s click reads as a rattle). GPU when present, CPU otherwise. The library pass resumes on unchanged size/mtime and skips unreadable files with a note.
