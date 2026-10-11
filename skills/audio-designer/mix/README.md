# Mix checker

CPU-only, read-only measurements against ../references/mix-plan.md. Requires
Python 3.10+, numpy, ffmpeg and ffprobe, already installed. No pip install,
network, recognition models, GPU work, or audio output from the checker.

Run from the audio-designer skill folder:

~~~powershell
python mix/mix_check.py check
python mix/mix_check.py stems --voice "C:\Audio\voice.wav" --music "C:\Audio\music.wav" --bed "C:\Audio\bed.wav" --sfx "C:\Audio\call.wav" "C:\Audio\stinger.wav" --mix "C:\Audio\final.wav"
python mix/mix_check.py report --voice "C:\Audio\voice.wav" --music "C:\Audio\music.wav" --bed "C:\Audio\bed.wav" --mix "C:\Audio\final.wav"
~~~

--bed, --sfx and --mix are optional. --sfx accepts multiple files and can be
repeated. Arguments are passed directly to subprocesses, without a shell.
Quote paths containing spaces; Windows drive paths and Unicode work.

Reuse exactly the same measurements without metering again:

~~~powershell
python mix/mix_check.py stems --voice "C:\Audio\voice.wav" --music "C:\Audio\music.wav" --json > measurements.json
python mix/mix_check.py report --input measurements.json > cue-sheet.md
~~~

report --input - reads measurement JSON from stdin. Plain report accepts the
same stem arguments as stems and meters them directly. The Markdown table uses
the plan's exact eight cue-sheet columns. Gain/EQ are unknown because they
cannot be reconstructed from supplied files; cue times are relative to the
supplied stem, starting at zero.

Exit codes: 0 = no FAIL (WARNs can remain); 2 = at least one measured FAIL,
with the full report still printed; 1 = decoding, metering or other input error.
Argument syntax errors use argparse's exit code 2. check returns 0 only when
every self-test passes, otherwise 1.

## Inputs and timeline

Supply the edited stems AFTER gain, ducking and EQ, already aligned at time
zero. An effect that starts later needs its leading silence included. The tool
does not align, offset, loop, pad, mix or normalise inputs. It never changes an
input file. Final mix must be supplied separately; it is not reconstructed.

Voice duration defines the comparison timeline. Music/bed comparisons use only
actual shared samples, including a correctly weighted last partial frame.
Short continuous stems fail coverage beyond 0.1 s tolerance, rather than being
silently padded. Longer tails are WARN; they are fully metered for LUFS/peak
but excluded from voice-window comparisons. Effect files may end earlier.

## What the measurements mean

All assessment thresholds are in THRESHOLDS at the top of mix_check.py, with
source line numbers in mix-plan.md. The plan says these are starting points,
not standards, and project profiles take precedence. This checker uses that
default plan; it does not load a project profile. Local tolerances needed to
turn "about", "ducked further" and "lower still" into statuses are explicitly
labelled policy, not attributed to the plan as exact numbers. Gap assessments
use displayed 0.1 dB precision to avoid a rounded 30.0 receiving a boundary WARN.

| Measurement | Meaning / assessment |
|---|---|
| integrated | Whole supplied file, FFmpeg ebur128 integrated LUFS, with its normal gates. Voice/final: target -16, PASS within 1 LU, WARN within 2, FAIL beyond. Music/bed/effects have no absolute target, so their LUFS are INFO; the relative gaps are judged. |
| peak | Whole-file oversampled true peak from ebur128=peak=true, in dBTP. Stems: PASS <=-1.5; WARN above -1.5 but below 0; FAIL >=0. Final: PASS <=-1.5; WARN >-1.5 through -1.0; FAIL >-1.0. Lower peaks pass; the ceiling is not a peak to aim for. |
| gate / pauses | Seconds of voice-stem activity/inactivity. Non-overlapping 100 ms K-weighted mean channel-power frames; threshold is max(-55 dB, 95th percentile of nonzero frame levels minus 18 dB). Need 0.4 s activity; otherwise voice gate FAIL. Need 0.4 s pauses to measure ducking. |
| gate_threshold / active_level | Gate threshold and active voice K-weighted level. INFO: the plan sets no absolute target. |
| level_on / level_off | Layer K-weighted level in the shared active/pause windows. Energy and sample counts are pooled before conversion to dB. INFO. |
| gap_on | Voice minus layer, K-weighted (the LUFS filter), during active voice windows, in LU. Positive means the layer is quieter. Music: PASS 6-20; WARN 4-<6 or >20; FAIL <4. Bed: PASS >=30; WARN 26-<30; FAIL <26. Calibrated on two narrated shorts a listener approved (music 7.6 and 7.4, beds 35.5 and 31.9). |
| gap_off | ACTIVE voice reference minus music in pauses. INFO; compare with gap_on. |
| duck | Music level_off minus level_on. INFO: the gap during speech is the requirement and ducking one way to hold it; the approved shorts dip about 3 LU. Unavailable without usable pauses. |
| brightness | Bed spectral energy above 3.5 kHz (unweighted), as a percentage. INFO: K-weighting already counts the highs louder, and an approved cricket bed (77%) sat 32 LU under. |
| lra | Bed whole-file EBU loudness range in LU. WARN >=18, otherwise PASS. Short clips can substantially understate bursts; this is not a burst locator. |
| coverage | Actual common music/bed duration divided by voice duration. Missing more than 0.1 s fails; excess tail warns. Missing content inside an otherwise full-length file is not detected by coverage. |
| peak_gap | Voice true peak minus each effect's true peak. PASS 6-14 dB; WARN 3-<6 or >14; FAIL <3. An approved owl call: 9.0; a reference monkey call: 7.2. |
| overlap | Seconds of effect energy-gate activity overlapping voice activity. FAIL beyond one 0.1 s frame (the gate's edge). Game/UI sounds may intentionally overlap; pass only calls and stingers here. |

INFO is a reference measurement with no target. WARN is a borderline result or
an unavailable comparison. Read the explanation in stems; it names
the reason and plan source for every metric. No unsupported absolute target is
invented for music, bed or effect integrated LUFS. Every measured number in the
cue-sheet table carries its status; cue timing is metadata, not a level test.

## Known limits

- The energy gate is NOT a voice detector. Noise, breaths, sustained tones and
  leaked music can open it; quiet words can be missed. Its short frames are an
  energy gate, not EBU short-term LUFS. It has no speech recognition.
- Relative gaps are K-weighted (BS.1770 filter, ungated), not differences of
  integrated LUFS and not a model of perceptual masking. Frequency carving, actual intelligibility and listening
  remain pending. Summing channel powers avoids antiphase stereo cancellation.
- A measured inside/outside difference is evidence consistent with ducking.
  Natural music dynamics or arrangement changes can produce the same number.
  Compression ratio, attack and release cannot be inferred. Pause edges/release
  tails and the 100 ms grid can bias the comparison.
- Integrated LUFS uses the WHOLE stem and FFmpeg's EBU gating, while relative
  gaps use only active voice windows and no EBU gating. They can disagree. EBU values at/below its -70 LUFS absolute gate
  are shown as n/a/WARN, never as a trusted -70 measurement. Silence/non-finite
  comparisons never turn into infinite "good" gaps.
- Decoding resamples analysis to 48 kHz and loads one whole stem at a time into
  memory. Very long stems need corresponding RAM. Original files, not the
  resampled analysis buffer, supply the whole-file LUFS/true-peak meters.
  Supply single-audio-stream files; meters select the first audio stream.
- Silence within a music file, unpadded sidechain truncation within the file,
  section jumps, source-pack/biome coherence, labels, licensing and captions
  are outside this checker. It does not run audio_tag.py or audio_map.analyze.
- It does not establish that --mix contains the supplied stems or that supplied
  gains match a cue sheet. It only meters the supplied final file.
- A project profile or platform requirement can override the starting point.
  Listening is always pending, including on a completely passing report.

## Existing helpers and isolation

The checker imports audio_map.py's media lookup, loader, loudness,
loudness_samples and band_shares; no decoder or FFmpeg meter is duplicated.
Its loader currently imports soundfile unconditionally. A temporary stdlib
module adapter selects that loader's existing FFmpeg fallback, then restores
sys.modules immediately. No soundfile dependency or shared-file edit is needed.
The checker is single-threaded; the adapter is not suitable for concurrent
library calls in the same Python interpreter.

audio_map.media_tool handles FFMPEG, the WinGet Gyan installation and PATH.
FFMPEG, when needed, is the absolute ffmpeg executable path; ffprobe must be
beside it. Existing helper FFmpeg calls use one decoder thread. No hardware
acceleration or model code is invoked. Bytecode writes are disabled before
shared imports, so the scripts folder is left untouched.

check generates deterministic sine/noise activity with gaps, music with known
levels, ducked/unducked music, bright/dark beds, loud bed/master, silence,
continuous activity, truncated music, antiphase stereo and timed effects in a
temporary directory. It asserts both correct measurements and rejection of
broken inputs, exercises CLI exit codes and the Markdown report, and removes
all generated audio. No real library material is stored in this folder.

