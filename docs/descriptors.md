# Descriptor reference

`skills/audio-designer/scripts/audio_describe.py` measures every audio file in a
library without a model and writes one row per file to `AUDIO_DESCRIPTORS.csv`
in the library root. The pass resumes on unchanged size and modification time,
and measures a file again when the `describer` version changes. Decoding uses the
file's own sample rate and channels; one-channel measures use the channel mean.

Every measure below is a measurement of the samples. None of them says what a
sound *is*: "punchy" describes an attack and a crest factor, not a punch.

## Measures

| Column | Unit | How it is computed | Known limits |
|---|---|---|---|
| `duration_s` | s | Container duration (soundfile), else decoded length | Files over 10 minutes: only the first 10 minutes are decoded, plus the last 2 s for the tail and loop tests (`analysed_s` says how much). |
| `lead_silence_s`, `tail_silence_s` | s | Samples before the first and after the last sample above the silence floor: −60 dBFS or 50 dB below the file's peak, whichever is higher | A noisy recording has no silence by this rule; a very quiet fade-out below the floor counts as silence. |
| `sound_s` | s | `duration_s` minus both silences | |
| `true_peak_dbtp` | dBTP | 4× polyphase oversampling (`scipy.signal.resample_poly`) of the 4096-sample blocks holding samples within 6 dB of the sample peak, maximum absolute value | Equivalent to BS.1770 Annex 2 in intent; not a certified meter. |
| `sample_peak_dbfs` | dBFS | Maximum absolute sample | |
| `crest_db` | dB | Sample peak minus RMS of the trimmed sound (first 30 s) | Long sounds with one loud event read as high crest. |
| `lufs_integrated` | LUFS | ITU-R BS.1770-4: K-weighting (libebur128's biquad design, equal to the tabulated 48 kHz coefficients), 400 ms blocks with 75 % overlap, −70 LUFS absolute and −10 LU relative gates, channels summed with weight 1 | **Blank under 400 ms**: the standard defines no block. Surround channel weights are not applied. Checked against FFmpeg `ebur128` within 0.1 LU. |
| `lufs_short_max` | LUFS | Maximum of a 3 s sliding window (100 ms hop) | A file shorter than 3 s is metered as a meter would: padded with silence, so a 0.2 s click reads low. |
| `lufs_momentary_max` | LUFS | Maximum of a 400 ms sliding window | The loudness word uses this one, as it suits short sounds. |
| `attack_ms` | ms | Hilbert envelope smoothed over 1 ms; the first event reaching half the loudest one (10 ms smoothing); time from 10 % to 90 % of that event's peak | A 5 ms linear ramp measures 4 ms (by definition, 80 % of it). Sounds that swell over several events read the first one. |
| `decay_ms` | ms | From the loudest moment (10 ms smoothing) until the envelope first falls 20 dB below it | **Blank** when it never does within the analysed span: sustained sounds and loops. |
| `centroid_hz`, `rolloff_hz` | Hz | `librosa.stft` (2048 / 512) power spectrum; per-frame centroid and 85 % rolloff, averaged with each frame's energy as weight | Brightness words use the centroid's percentile *within the library*, never absolute Hz. |
| `flatness` | 0–1 | `librosa.feature.spectral_flatness` on the power spectrum, energy-weighted | Real recordings sit far below white noise (~0.5): band-limited noise reads low without having a pitch, which is why `tonal` uses the pitch instead. Library median 0.0007, top quarter ≥ 0.0076; noise-named files median 0.0032, tone-named 0.0003. |
| `low_share` | 0–1 | Share of spectral energy below 150 Hz | |
| `pitch_hz` | Hz | `librosa.pyin` (40–2000 Hz) on the first 2 s of the sound at 22.05 kHz; median f0 of frames with voicing probability ≥ 0.5, reported when the mean voicing probability is ≥ 0.5. Runs only when flatness ≤ 0.1. | **Blank** when unvoiced. Strict on purpose: single-pitch tones (beeps, sine-like UI tones, sung notes) pass; chords, bells, piano music and fast sweeps mostly do not. On 80 files named beep/tone/chime/bell/note/piano, 6 had a pitch; on 80 named noise/hiss/wind/rain, 1. |
| `onsets` | count | Peaks of `librosa.onset.onset_strength` reaching 30 % of the strongest, at least 50 ms apart, on the sound with 50 ms of silence before and 100 ms after | Events closer than 50 ms merge; noisy textures (wind, crowds) give irregular counts. librosa's default picker counted a resonant hit as two or three events: on 80 short files named impact/hit/click this rule calls 78 % single (default 52 %) and still calls 85 % of 20 rattles, steps and bursts repeated (default 95 %). |
| `loop_ready` | 0/1 | Files of 1 s or more only. Both 50 ms ends within 3 dB of each other and within 12 dB of the body, log-band spectra with cosine ≥ 0.95, and the jump from the last sample to the first no larger than 4× the median sample-to-sample step at the ends | A test of the seam, not of musical phrasing: a loop that is seamless but rhythmically wrong passes. |
| `clipped_runs` | count | Runs of 3 or more consecutive samples at ≥ 99.9 % of full scale, per channel | Clipping already reduced by a limiter is not detected. |

Validation (`python audio_describe.py check`) generates signals with known
answers: a 997 Hz full-scale sine reads −3.01 LUFS; a 440 Hz tone reads 440 Hz;
white noise is flat and unpitched; band-limited noise is unpitched; a 5 ms ramp
reads ~4 ms attack; a 20 dB-in-300 ms exponential reads ~300 ms decay; 50 ms of
leading silence is reported; a clipped sine is flagged; a whole-cycle loop
passes and the same loop cut mid-cycle, fading, or fading out and in fails;
one click counts 1 onset and five count 5.

## Plain words

`WORDS` in `audio_describe.py` is the only place a word is defined. Each row is
the word, its aliases, the conditions (all must hold) and one sentence the
browser shows in Explain mode.

| Word | Aliases | Condition |
|---|---|---|
| short | brief | `sound_s` < 0.5 |
| long | lengthy | `sound_s` ≥ 3 |
| punchy | snappy, tight | `attack_ms` ≤ 10 and `crest_db` ≥ 12 |
| soft | | `attack_ms` ≥ 40 |
| bright | | centroid percentile ≥ 0.75 |
| dark | muffled | centroid percentile ≤ 0.25 |
| deep | bassy, boomy | `low_share` ≥ 0.5 |
| tonal | pitched | a pitch was found |
| noisy | hissy | `flatness` ≥ 0.01 |
| single | one-shot, oneshot | `onsets` = 1 |
| repeated | | `onsets` ≥ 3 |
| dry | | `decay_ms` ≤ 200 |
| tail | reverby | `decay_ms` ≥ 1000 |
| loud | | `lufs_momentary_max` ≥ −14 |
| quiet | | `lufs_momentary_max` ≤ −30 |
| loopable | seamless | `loop_ready` = 1 |

In a search, every word or alias becomes a filter and is removed from the text
the search ranks: `short punchy bright click` ranks "click" by the fused search,
puts the files that are short, punchy and bright first (marked as matching), and
lists the rest after them. Without `AUDIO_DESCRIPTORS.csv` the words stay in the
search text, as before.

**To change a word**, edit its row: a threshold, a measure or a new alias. A
percentile measure ends in `_pct` and is computed when the library is read, so
it needs no new pass. A new measure needs a column in `FIELDS`, a line in
`measure()`, a known-answer test in `check()`, and a `VERSION` bump so every
file is measured again.

## Cost

Decoding, K-weighting and spectral analysis dominate the pass. Cost varies with
file length, format and hardware; `--workers` sets the parallel worker count.
