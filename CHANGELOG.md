# Changelog

Changes are grouped by capability, newest first within each group. Dates and commit IDs come from the repository history. Each release lists what it adds on top of the dated history below.

## 1.0.0 · 2026-10-15

First tagged release. It includes everything in the dated history below, plus:

- Browser: cached waveform previews, a seekable playhead, spectrograms, keyboard review and side-by-side loudness-matched comparisons.
- Getting started: a generated demo library, browser screenshots and complete command-line help.
- Documentation: a measurement-and-evidence walkthrough and user-facing category limits.
- Categories: UCS search, aliases, a model-free source catalogue and local Y/N checks.
- Reliability: explicit unreadable, short and silent rows; separate caches for read-only libraries.
- Publication: research tools, private benchmarks and datasets stay local.

## Search and library browsing

- **2026-10-12 · `8c1f291`** — Collapse duplicate copies into one search-result row.
- **2026-10-12 · `e55a713`** — Add measured descriptive words, duplicate groups, variation sets and effect export.
- **2026-10-12 · `67a7ca9`** — Add full-file library metadata, a fresh-copy smoke test and retrieval from saved approval and rejection verdicts.
- **2026-10-11 · `b0cf5fe`** — Search by approved and rejected audio examples.
- **2026-10-11 · `d6f5c99`** — Show canonical tags with agreement between name, model labels and audio evidence.
- **2026-10-11 · `e722263`** — Fuse independently ranked audio, cleaned names and label evidence.
- **2026-10-10 · `461a86f`** — Introduce the audio toolkit, tagger and local sound browser.

## Measurement, recognition and evaluation

- **2026-10-12 · `e55a713`** — Measure envelope, brightness, pitch, loudness, silence, onsets and loop seams; add short-effect benchmarks and game-oriented vocabulary.
- **2026-10-12 · `67a7ca9`** — Preserve full-duration metadata in a resumable ledger and record skipped-file reasons.
- **2026-10-11 · `3416bf3`** — Correct full-library tagging behavior and add mix measurements.
- **2026-10-11 · `3fdea02`** — Fit per-tag label rules using held-out packs.
- **2026-10-11 · `3be768b`** — Adopt CED-base for AudioSet labels and silence-pad short CLAP fingerprints.

## Rendering and mix checks

- **2026-10-12 · `67a7ca9`** — Calibrate mix thresholds from approved aligned stems; add crossfaded cue loops and peak alignment.
- **2026-10-11 · `b0cf5fe`** — Warn on short ambience where a reliable measurement is unavailable.
- **2026-10-11 · `d2b52b0`** — Render cue sheets to aligned stems and a checked final mix; calibrate mix checking against approved mixes.
- **2026-10-11 · `3416bf3`** — Introduce the stem and final-mix checker.

## Distribution and documentation

- **2026-10-12 · `67a7ca9`** — Add portable launchers, pinned dependencies, privacy checks and one test runner for source and publication copies.
- **2026-10-12 · `ee9ac65`** — Update renderer documentation and use consistent text-file line endings.
- **2026-10-11 · `b4d6818`** — Refuse publication when the staged copy fails self-tests.
- **2026-10-11 · `b3c8b1d`** — Restore the finished tagging self-test.
- **2026-10-11 · `bc8e828`** — Package launchers, requirements and the first project README.
- **2026-10-10 · `4f08d18`** — Create the repository.
