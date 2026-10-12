# Cue-sheet mix renderer

CPU only: Python 3.10+, the existing numpy, ffmpeg and ffprobe. No installs,
network calls, recognition models, GPU or playback. Run from the skill folder:

~~~powershell
python -B render/render_mix.py render "episode/cues.csv" --out "episode/rendered"
python -B render/render_mix.py render "episode/cues.csv" --out "episode/rendered-custom" --levels levels.json
python -B render/render_mix.py check
~~~

The shared audio_map.media_tool lookup uses FFMPEG (an executable path, with
ffprobe beside it), the WinGet Gyan bundle, then PATH. Set FFMPEG for the current
shell if needed. No personal executable/library paths are built into the renderer.

The output must be a new or empty directory. Existing files are never overwritten.
Inputs are read only; relative file paths resolve beside the CSV, not beside the
script. Local Windows paths with spaces, commas and non-ASCII names work; quote
CSV cells containing commas. CSV is UTF-8, with an optional BOM. Lines beginning
with `#` (optionally after whitespace) are comments, including before the header. The shared
audio_map.probe helper explicitly decodes ffprobe output as UTF-8.

## CSV

The plan's cue-sheet fields are split into numeric columns for editing. Times
are seconds, decimal points are dots, and all timing rounds to 48 kHz sample
frames (one frame = one sample per channel). Here is one complete example:

~~~csv
layer,file,excerpt_start,excerpt_end,cue_in,cue_out,gain_db,duck_ratio,highpass_hz,lowpass_hz,fade_in,fade_out,loop,align
voice,voice.wav,0,10,0,12,0,-,-,-,0.01,0.02,no,start
music,music.wav,2,6,0,12,-18,3:1,-,12000,0.02,0.30,yes,start
bed,ambience.wav,0,4,0,12,-32,8:1,140,6000,0.02,0.30,yes,start
sfx,"UI/click, café.wav",0,0.2,4.5,,-12,-,-,-,0.005,0.02,no,peak
~~~

These gains are examples, not a promise that arbitrary sources pass the checker.

| Column | Meaning / omitted or blank value |
|---|---|
| layer | Required: voice, music, bed or sfx (case insensitive). Multiple rows may share a layer and overlap; their samples sum without automatic normalisation. |
| file | Required local audio file. First audio stream, mono or stereo. Mono is duplicated at unchanged amplitude into stereo. |
| excerpt_start | Source position to begin, after resampling to 48 kHz; default 0. |
| excerpt_end | Exclusive source end; default source duration. Must follow start and fit the source. |
| cue_in | Timeline position for the excerpt's first sample; default 0. With sfx align=peak, this is the peak's timeline position. |
| cue_out | Exclusive timeline end. Default is the placed excerpt end, accounting for peak alignment. Earlier ends trim audio; later ends add silence unless loop=yes. |
| gain_db | Fixed gain before the sum; default 0. Negative values attenuate. Never auto-adjusts a source to pass. |
| duck_ratio | Music/bed only: 1 through 20, optionally N:1; blank or '-' disables sidechain for that row. Typical plan starts are music 4:1–8:1, bed 8:1. |
| highpass_hz | Optional two-pole FFmpeg high-pass; blank/'-' disables. 140 is a bed starting point. |
| lowpass_hz | Optional two-pole FFmpeg low-pass; blank/'-' disables. Cutoffs must be >0, <24000, and HP below LP. |
| fade_in | Linear fade duration at the excerpt's audible start; default 0.01. Explicit 0 disables. |
| fade_out | Linear fade duration at the audible end after trimming; default 0.02. Explicit 0 disables. Fades together must fit the audible cue. |
| loop | Music/bed only: yes repeats a short excerpt through cue_out, with a linear crossfade of up to 20 ms (at most half the excerpt) at each join. No, blank or omitted retains silence padding after the source ends. No time stretching. |
| align | start (default), or peak for sfx. Peak places the first largest absolute sample across either source channel, within the selected excerpt, at cue_in. It is measured before gain/EQ/fades. |

For a peak-aligned effect whose onset would fall before time zero, the leading
samples are cropped while the peak stays on cue_in. Omitting cue_out keeps the
remaining excerpt tail; an explicit cue_out trims or pads the placed effect.
Fades remain at the original excerpt's edges before cropping. EQ and fades can
change which sample is loudest after processing; inspect the rendered effect.

The plan's combined table fields also work as CSV headers: Layer, File, Excerpt,
Cue in/out, Gain, Duck / EQ, Measured after gain, Listening, with an optional
Fade in/out. Use Excerpt = whole or 2-4 s; Cue in/out = 5-7 or a single start;
Duck / EQ = "HP 140 Hz, LP 6000 Hz, duck 8:1"; Fade in/out = 0.01/0.02.
Measured after gain and Listening are accepted input metadata and ignored; new
measurements come from the checker. Use sfx for the plan's Call/Effect role.
Separate duck ratio and EQ columns are also accepted; EQ = "140/6000" or "-/6000".
Choose either combined or split timing fields; combined values take precedence.
Header spaces, punctuation and case are normalised; unknown/duplicate columns fail.

## Processing and output

Gain, excerpts, EQ and fades are baked into voice.wav, music.wav, bed.wav and
sfx.wav. All four stems start at zero and have the same duration: the latest
cue_out. Missing layers become silence. Output is stereo 48 kHz float32 WAV:
float preserves samples so a clipped sum remains detectable by the checker.

Voice renders first. Each music/bed cue with a ratio uses sidechaincompress
keyed by the gained voice, with a key explicitly padded through the full
timeline. Ducking is included in those stems. DUCK at the top of the script
copies the cutter of the approved narrated shorts: music threshold 3.8 dB under
the voice's integrated LUFS, attack 20 ms, release 400 ms; beds 13.3 dB under,
10 ms, 350 ms. The ratio comes from the CSV (the approved shorts use 3 for music,
8 for beds). Re-rendering one approved short's voice and music this way measured the same as
the original (music 7.4 LU under the voice, a 2.8 LU dip). MIX_START holds the
master targets: -16 LUFS and -1.5 dBTP from mix-plan.md:43, LRA=11 a local choice.

`--levels levels.json` loads the policy written by
`mix/mix_check.py calibrate --stems APPROVED_DIR [APPROVED_DIR ...] --out levels.json`.
The renderer uses its checker thresholds and `ducking` settings and records the
effective policy in render.json. The ducking JSON has music/bed entries with
threshold_db_re_voice, attack_ms and release_ms; ratios still come from each CSV
row. Calibration measures gaps from baked stems and preserves the default
compressor settings, since those settings cannot be recovered from the audio.
Edit those JSON settings when needed. Without --levels, thresholds and ducking
remain the original defaults.

The exact four stems sum with amix=normalize=0. One whole-mix loudnorm operation
uses two passes: measure, then apply those measurements. FFmpeg can fall back
to dynamic normalisation when linear gain cannot satisfy peak/LRA constraints.
Output is resampled and bounded again at 48 kHz for exact alignment. Stems are
before final loudnorm: the final may change their overall gain/dynamics.

render.json records resolved cue settings, exact argument arrays for every
FFmpeg/ffprobe render and shared decoder/meter command, both loudnorm JSON
measurements, the imported checker's full metrics and exit code, and SHA-256
identities of the shared checker/helpers/plan. Commands use
argument arrays, never a shell. cue_report.md contains mix_check's original
report rows and rules. Its reconstructed gain/EQ cells say unknown; the exact
cue gains/EQ are in the CSV and render.json.

Exit 0 = no checker FAIL (WARN/INFO may remain); 2 = checker FAIL, with all
outputs/reports retained; 1 = input/render/meter error. On a rendering error
after output creation, render.json records the error and commands completed.
Invalid CSV is rejected before output creation.

## Verification and limits

check generates deterministic inputs in a system temporary directory
and removes them afterwards. It renders the correct implementation and six
in-memory source mutations: no ducking, lost cue delay, wrong excerpt, ignored
gain, -10 LUFS master, and missing alignment. Each of the six corresponding
audio assertions must reject its mutation; a crash does not count as a rejection.
The unducked fixture must also make the real mix_check return a music FAIL and
propagate exit 2. The fixture is deliberately loud enough to fail the speech
gap when ducking is removed, including checker versions that report duck depth
as INFO. Correlation expectations derive from source samples, not render helpers.
It also checks a passing report, Unicode/comma paths and an audible padded tail.
Three more source mutations remove looping, remove crossfades, and ignore peak
alignment. The corresponding sample assertions reject silence after a short
source, an abrupt loop seam, and a misplaced negative stereo peak. Additional
checks cover pre-zero cropping, unchanged behavior with omitted columns,
invalid column values, stricter --levels failures, and changed ducking samples.

No source identity, licensing, voice intelligibility, spectral taste, source-pack
coherence or peak-to-visual contact is inferred. Synthesised "voice" is only an
energy-gate fixture, not a voice-detector validation. The checker's 100 ms gate
can miss words or misread noise; its own policy is used unchanged and saved in
the report. A voiceless cue sheet produces a silent-voice checker FAIL. UI cues
use the checker's sfx pause rule, so intentional speech overlap may fail.

Source durations rely on ffprobe metadata (compressed-file boundaries may be
approximate). Inputs with more than two channels are rejected. Large timelines
need substantial disk/RAM: the imported checker decodes whole stems. No section
jump/silence scan, automatic gain fixing, music presence guarantee,
parameter automation, recognition pass or project-profile override is included.
Repeated cues on the same layer sum and may clip; inspect FAILs before listening.
Subprocess logging and the existing decoder adapter require single-threaded use.


The published self-test generates its own synthetic audio and needs no private library or production demo.
