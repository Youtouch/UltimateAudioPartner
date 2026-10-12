"""Measure what every sound is like, without a model: length, loudness, attack, decay, brightness, pitch, loops.

    python audio_describe.py describe --root D:/MySounds        # library pass -> AUDIO_DESCRIPTORS.csv, resumable
    python audio_describe.py show FILE [FILE ...]               # one file's measures and plain words, JSON
    python audio_describe.py check                              # synthetic signals with known answers

Every number is a measurement of the samples (definitions, units and limits: docs/descriptors.md in the
repository). The plain words a game developer types ("short punchy bright") are defined in WORDS below, one
row per word, each naming the measure behind it: edit that table to change what a word means.
"""

import argparse
import csv
import fnmatch
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from audio_map import load_audio
from library_metadata import relative_path

VERSION = 'v1'  # bump when a measure changes: rows written by another version are measured again
FIELDS = ['relative_path', 'size_bytes', 'mtime_ns', 'duration_s', 'lead_silence_s', 'tail_silence_s', 'sound_s',
          'analysed_s', 'channels', 'sample_rate', 'true_peak_dbtp', 'sample_peak_dbfs', 'crest_db', 'lufs_integrated',
          'lufs_short_max', 'lufs_momentary_max', 'attack_ms', 'decay_ms', 'centroid_hz', 'rolloff_hz', 'flatness',
          'low_share', 'pitch_hz', 'onsets', 'loop_ready', 'clipped_runs', 'describer']
AUDIO = {'.wav', '.mp3', '.ogg', '.flac', '.aif', '.aiff', '.m4a', '.aac', '.opus', '.mp4', '.mov'}
SILENCE_DB = -60.0      # trim floor in dBFS ...
SILENCE_BELOW_PEAK = 50.0  # ... or this far below the file's peak, whichever is higher
DETAIL_S = 30.0         # attack, decay, spectrum and onsets read the first 30 s of the sound
PITCH_S = 2.0           # pitch reads the first 2 s of the sound (pYIN is the slowest measure)
PITCH_MAX_FLATNESS = 0.1  # pYIN runs only on sounds this tonal or more; noise has no pitch to find
MAX_DECODE_S = 600.0    # longer files: the first 10 minutes, plus their last 2 s for the tail and loop checks
CLIP_LEVEL, CLIP_RUN = 0.999, 3  # clipping: at least 3 consecutive samples at or above 99.9% of full scale

# ---------------------------------------------------------------------------------------------------------
# Plain words. One row per word: the measure, the test, and one sentence for the browser's Explain mode.
# A percentile measure (*_pct) is the file's rank within this library, 0 = lowest, 1 = highest.
# Thresholds were checked on a 35k-file game and media library (docs/descriptors.md). Energy-weighted flatness is
# far below white noise's ~0.5 for real recordings: 0.01 is that library's top quarter, where noise-named files sit.
WORDS = [
    # word,       aliases,                              [(measure, op, value), ...] all must hold,  explain
    ('short',     ('brief',),                    [('sound_s', '<', 0.5)],
     'The sound itself lasts under half a second once silence is trimmed: it fits a quick game event.'),
    ('long',      ('lengthy',),                          [('sound_s', '>=', 3.0)],
     'The sound lasts 3 seconds or more: a bed, a build-up or a slow event rather than a hit.'),
    ('punchy',    ('snappy', 'tight'),                   [('attack_ms', '<=', 10.0), ('crest_db', '>=', 12.0)],
     'It reaches full level within 10 ms and its peak stands well above its average: it hits hard and reads instantly.'),
    ('soft',      (),                  [('attack_ms', '>=', 40.0)],
     'It fades in over 40 ms or more: it arrives gently and will not startle.'),
    ('bright',    (),                    [('centroid_pct', '>=', 0.75)],
     'Its energy sits higher than in three quarters of this library: it cuts through a mix and reads as light or sparkly.'),
    ('dark',      ('muffled',),          [('centroid_pct', '<=', 0.25)],
     'Its energy sits lower than in three quarters of this library: it feels heavy, distant or muffled.'),
    ('deep',      ('bassy', 'boomy'),             [('low_share', '>=', 0.5)],
     'Half or more of its energy is below 150 Hz: you feel it as weight; phone speakers barely play it.'),
    ('tonal',     ('pitched',),                [('pitch_hz', '>', 0)],
     'It has a clear pitch or tone rather than a noise: it can clash or harmonise with the music.'),
    ('noisy',     ('hissy',),                   [('flatness', '>=', 0.01)],
     'It is mostly noise, like hiss, wind or a scrape: no single pitch.'),
    ('single',    ('one-shot', 'oneshot'),               [('onsets', '==', 1)],
     'One event, not several: a single hit, click or note.'),
    ('repeated',  (),              [('onsets', '>=', 3)],
     'Three or more separate events in a row: a rattle, a run of steps or a burst.'),
    ('dry',       (),                            [('decay_ms', '<=', 200.0)],
     'It dies away within 0.2 s of its peak: no echo or tail, so it stays crisp when played quickly in a row.'),
    ('tail',      ('reverby',), [('decay_ms', '>=', 1000.0)],
     'It keeps ringing for a second or more after its peak: reverb or resonance, which can smear when repeated fast.'),
    ('loud',      (),                                    [('lufs_momentary_max', '>=', -14.0)],
     'Its loudest 0.4 s reaches -14 LUFS or more: already mastered loud, little gain needed.'),
    ('quiet',     (),                   [('lufs_momentary_max', '<=', -30.0)],
     'Its loudest 0.4 s stays at or below -30 LUFS: quiet as recorded, it will need gain.'),
    ('loopable',  ('seamless',),       [('loop_ready', '==', 1)],
     'Its start and end match in level, tone and waveform: it should repeat without an audible seam.'),
]
# Aliases are few on purpose: on the reviewed-clip benchmark, everyday adjectives as filters ('quick sweeping
# whoosh', 'steady gentle rain', 'a single soft wooden click') removed relevant files whose writer meant a feel, not a measure.
WORD_INDEX = {alias: row[0] for row in WORDS for alias in (row[0], *row[1])}
# One sentence per number, for the browser's Explain mode: what the measure means for a sound in a game.
MEASURE_EXPLAIN = {
    'sound_s': 'How long the sound itself lasts, silences excluded. Under 0.5 s suits quick, frequent events.',
    'attack_ms': 'How fast it reaches full level. Under 10 ms feels instant and punchy; over 50 ms swells in softly.',
    'crest_db': 'How far its peak stands above its average. High (12 dB or more) reads as a sharp hit; low reads as dense or sustained.',
    'centroid_pct': 'Where its energy sits compared with the rest of your library: high is bright and cutting, low is dark and heavy.',
    'low_share': 'How much of its energy is deep bass (below 150 Hz). Lots of it adds weight, but phone speakers will not play it.',
    'flatness': 'Noise (near 1) versus tone (near 0). Noisy sounds sit behind music; tonal ones can clash with its key.',
    'onsets': 'How many separate events it contains: 1 is a single hit, several is a rattle, a run or a burst.',
    'decay_ms': 'How long it rings after its peak until it is 20 dB quieter. Short is dry and tight; long has a tail or reverb.',
    'lufs_momentary_max': 'Its loudest moment, in LUFS (the loudness unit stores and games use). Around -14 is loud; -30 is quiet.',
    'loop_ready': 'Whether its end flows back into its start without a seam, so it can repeat as a loop.',
    'pitch_hz': 'Its musical pitch, when it has one. Lower numbers sound deeper.',
    'true_peak_dbtp': 'Its highest peak, including between samples. Above -1 dBTP it may distort once compressed for a game build.',
}
OPS = {'<': np.less, '<=': np.less_equal, '>': np.greater, '>=': np.greater_equal, '==': np.equal}
NUMBERS = {  # how the browser prints a measure under a word: label, unit, decimals
    'sound_s': ('length', 's', 2), 'attack_ms': ('attack', 'ms', 1), 'crest_db': ('crest', 'dB', 1),
    'centroid_pct': ('brighter than', '% of this library', 0), 'centroid_hz': ('centroid', 'Hz', 0),
    'low_share': ('energy below 150 Hz', '%', 0), 'flatness': ('flatness', '', 3), 'onsets': ('events', '', 0),
    'decay_ms': ('decay to -20 dB', 'ms', 0), 'lufs_momentary_max': ('loudest 0.4 s', 'LUFS', 1),
    'loop_ready': ('loop seam', '', 0), 'pitch_hz': ('pitch', 'Hz', 0), 'true_peak_dbtp': ('true peak', 'dBTP', 1),
}


def split_query(text):
    """'short punchy bright click' -> ('click', ['short', 'punchy', 'bright']). Plain words become filters."""
    kept, filters = [], []
    for token in text.split():
        word = WORD_INDEX.get(token.strip('.,;:!?').lower())
        if word:
            if word not in filters:
                filters.append(word)
        else:
            kept.append(token)
    return ' '.join(kept), filters


def word_mask(word, table):
    """Boolean mask over the columns of `table` (name -> float array, NaN = unmeasured). NaN never passes."""
    conditions = next(row[2] for row in WORDS if row[0] == word)
    mask = None
    for measure, op, value in conditions:
        values = table[measure]
        with np.errstate(invalid='ignore'):
            ok = np.isfinite(values) & OPS[op](values, value)
        mask = ok if mask is None else mask & ok
    return mask


def percentile(values):
    """Rank within the library, 0..1; NaN stays NaN."""
    out = np.full(len(values), np.nan)
    ok = np.isfinite(values)
    if ok.any():
        ordered = np.sort(values[ok])
        out[ok] = np.searchsorted(ordered, values[ok], side='right') / ok.sum()
    return out


def read_descriptors(root):
    """Last row per path from AUDIO_DESCRIPTORS.csv, or {} when the library has not been described."""
    path = Path(root) / 'AUDIO_DESCRIPTORS.csv'
    if not path.exists():
        return {}
    with path.open(encoding='utf-8', newline='') as f:
        return {relative_path(r['relative_path']): r for r in csv.DictReader(f) if r.get('describer')}


def table_for(paths, descriptors):
    """Column arrays (float, NaN = unmeasured) for `paths`, with the library-relative percentiles added."""
    measures = {m for row in WORDS for m, _, _ in row[2]} | set(NUMBERS)
    table = {}
    for m in measures:
        if m.endswith('_pct'):
            continue
        table[m] = np.array([_float(descriptors.get(p, {}).get(m)) for p in paths])
    everything = np.array([_float(r.get('centroid_hz')) for r in descriptors.values()])
    ordered = np.sort(everything[np.isfinite(everything)])
    c = table['centroid_hz']
    table['centroid_pct'] = np.where(np.isfinite(c), np.searchsorted(ordered, c, side='right') / max(1, len(ordered)), np.nan)
    return table


def words_for(table, i):
    return [row[0] for row in WORDS if word_mask(row[0], {k: v[i:i + 1] for k, v in table.items()})[0]]


def _float(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------------------------------------
# Loudness, ITU-R BS.1770-4 / EBU R128, in numpy. K-weighting coefficients are derived for any sample rate.

def k_weighting(fs):
    """The BS.1770 pre-filter (high shelf, then RLB high-pass) as two biquads, designed for any rate the way
    libebur128 does; at 48 kHz they equal the coefficients the standard tabulates."""
    k = np.tan(np.pi * 1681.974450955533 / fs)
    q, vh = 0.7071752369554196, 10 ** (3.999843853973347 / 20)
    vb, a0 = vh ** 0.4996667741545416, 1 + k / q + k * k
    shelf = (np.array([vh + vb * k / q + k * k, 2 * (k * k - vh), vh - vb * k / q + k * k]) / a0,
             np.array([1, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]))
    k, q = np.tan(np.pi * 38.13547087602444 / fs), 0.5003270373238773
    a0 = 1 + k / q + k * k
    highpass = (np.array([1.0, -2.0, 1.0]), np.array([1, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]))
    return [shelf, highpass]


def loudness_curve(y, sr):
    """K-weighted power summed over channels, as a cumulative sum for fast windowed means."""
    from scipy.signal import lfilter
    power = np.zeros(len(y))
    for ch in range(y.shape[1]):
        x = y[:, ch].astype(np.float64)
        for b, a in k_weighting(sr):
            x = lfilter(b, a, x)
        power += x * x
    return np.concatenate([[0.0], np.cumsum(power)])


def _lufs(mean_power):
    with np.errstate(divide='ignore'):
        return -0.691 + 10 * np.log10(mean_power)


def window_max(cum, sr, seconds, hop=0.1):
    """Max loudness of a sliding window. A file shorter than the window is metered as a meter would: padded with silence."""
    size, n = int(round(seconds * sr)), len(cum) - 1
    if n <= size:
        return float(_lufs(cum[-1] / size))
    starts = np.arange(0, n - size + 1, max(1, int(hop * sr)))
    return float(_lufs(((cum[starts + size] - cum[starts]) / size).max()))


def integrated(cum, sr):
    """Gated integrated loudness (400 ms blocks, 75% overlap, -70 LUFS absolute and -10 LU relative gates).
    None under 400 ms, where the standard defines no block."""
    size, n = int(round(0.4 * sr)), len(cum) - 1
    if n < size:
        return None
    starts = np.arange(0, n - size + 1, max(1, size // 4))
    blocks = (cum[starts + size] - cum[starts]) / size
    blocks = blocks[_lufs(blocks) > -70]
    if not len(blocks):
        return None
    relative = _lufs(blocks.mean()) - 10
    kept = blocks[_lufs(blocks) > relative]
    return float(_lufs(kept.mean()))


def true_peak(y):
    """4x oversampled peak (BS.1770 Annex 2 style), in dBTP. Inter-sample peaks exceed the sample peak by a few dB
    at most, so only the 4096-sample blocks holding samples within 6 dB of it are oversampled."""
    from scipy.signal import resample_poly
    level = np.abs(y).max(axis=1)
    peak = float(level.max()) if len(level) else 0.0
    if peak <= 0:
        return -np.inf
    for block in np.unique(np.flatnonzero(level >= 0.5 * peak) // 4096):
        start, end = block * 4096, min(len(y), (block + 1) * 4096)
        lo, hi = min(64, start), min(64, len(y) - end)  # context samples, dropped after filtering (edge ringing)
        up = resample_poly(y[start - lo:end + hi], 4, 1, axis=0)
        peak = max(peak, float(np.abs(up[4 * lo:len(up) - 4 * hi]).max()))
    return _db(peak)


def _db(x):
    return float(20 * np.log10(x)) if x > 0 else -np.inf


# ---------------------------------------------------------------------------------------------------------
# Shape: trim, attack, decay, onsets, spectrum, pitch, loop seam, clipping.

def trim_points(mono, peak):
    """First and last sample above the silence threshold, or None for a silent file."""
    floor = max(10 ** (SILENCE_DB / 20), peak * 10 ** (-SILENCE_BELOW_PEAK / 20))
    above = np.flatnonzero(np.abs(mono) > floor)
    return (int(above[0]), int(above[-1])) if len(above) else None


def _smooth(x, n):
    n = max(1, min(int(n), len(x)))
    if n == 1:
        return x
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = (c[n:] - c[:-n]) / n
    return np.concatenate([np.full(n // 2, out[0]), out, np.full(len(x) - len(out) - n // 2, out[-1])])


def envelope(sound, sr):
    from scipy.signal import hilbert
    return np.abs(hilbert(sound.astype(np.float64)))


def attack_decay(sound, sr):
    """Attack: 10% to 90% of the first onset's peak. Decay: the loudest moment to 20 dB below it (None if it never gets there)."""
    env = envelope(sound, sr)
    fine, coarse = _smooth(env, sr * 0.001), _smooth(env, sr * 0.01)
    top = coarse.max()
    if top <= 0:
        return None, None
    first = int(np.argmax(coarse >= 0.5 * top))  # the first event that reaches half the loudest one
    window = fine[first:first + int(0.05 * sr) + 1]
    peak_at = first + int(np.argmax(window))
    peak = fine[peak_at]
    below = np.flatnonzero(fine[:peak_at] < 0.1 * peak)
    t10 = int(below[-1]) + 1 if len(below) else 0
    t90 = t10 + int(np.argmax(fine[t10:peak_at + 1] >= 0.9 * peak))
    attack = 1000 * (t90 - t10) / sr
    loudest = int(np.argmax(coarse))
    fallen = np.flatnonzero(coarse[loudest:] < 0.1 * top)
    decay = 1000 * fallen[0] / sr if len(fallen) else None
    return attack, decay


def spectrum(sound, sr):
    """Energy-weighted spectral centroid, 85% rolloff, flatness and the share of energy below 150 Hz."""
    import librosa
    n_fft = 2048
    sound = np.pad(sound, (0, max(0, n_fft - len(sound))))  # zero padding adds no energy to the weighted means
    s = np.abs(librosa.stft(sound.astype(np.float32), n_fft=n_fft, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    energy = s.sum(axis=0)
    if energy.sum() <= 0:
        return None
    w = energy / energy.sum()
    centroid = (freqs[:, None] * s).sum(axis=0) / np.maximum(energy, 1e-20)
    cumulative = np.cumsum(s, axis=0)
    rolloff = freqs[np.argmax(cumulative >= 0.85 * cumulative[-1:], axis=0)]
    flatness = librosa.feature.spectral_flatness(S=s, power=1.0)[0]  # S is already power
    low = s[freqs < 150].sum() / s.sum()
    return float((w * centroid).sum()), float((w * rolloff).sum()), float((w * flatness).sum()), float(low)


def pitch(sound, sr):
    """Median pYIN f0 over the first PITCH_S seconds, when the mean voicing probability is at least 0.5; else None."""
    import librosa
    from scipy.signal import resample_poly
    y = sound[:int(PITCH_S * sr)].astype(np.float64)
    if sr != 22050:
        y = resample_poly(y, 22050, sr)
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    f0, _, probability = librosa.pyin(y.astype(np.float32), fmin=40, fmax=2000, sr=22050, frame_length=2048)
    # pYIN's voiced flag also fires on narrow-band noise; its mean voicing probability does not (tones ~0.9, noise ~0.01).
    if np.nanmean(probability) < 0.5:
        return None
    return float(np.nanmedian(f0[probability >= 0.5]))


def onset_count(sound, sr):
    """Separate events: peaks of librosa's onset strength reaching 30% of the strongest, at least 50 ms apart.
    librosa's default picker counted a single resonant hit as 2-3 events; on 80 short files named impact/hit/click
    this rule calls 78% single (default 52%), and still calls 85% of rattles, steps and bursts repeated (default 95%)."""
    import librosa
    from scipy.signal import find_peaks
    # Silence around the trimmed sound, so a hit at its first sample still rises and its last one still falls.
    y = np.concatenate([np.zeros(int(0.05 * sr)), sound, np.zeros(int(0.1 * sr))]).astype(np.float32)
    strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    if strength.max() <= 0:
        return 0
    return len(find_peaks(strength, height=0.3 * strength.max(), distance=max(1, int(0.05 * sr / 512)))[0])


def loop_ready(mono, sr, head=None):
    """A file loops cleanly when its two ends match: level within 3 dB of each other and within 12 dB of the
    body (a sound that fades out and in again pulses rather than loops), spectra alike, and the jump from its
    last sample back to its first no bigger than the signal's own sample-to-sample steps."""
    head = mono if head is None else head
    body = float(np.sqrt(np.mean(head.astype(np.float64) ** 2)))
    if len(mono) < 1.0 * sr:  # under a second a "loop" is a repeated one-shot, whatever its seam
        return 0
    n = int(min(0.05 * sr, len(mono) // 4))
    a, b = head[:n].astype(np.float64), mono[-n:].astype(np.float64)
    ra, rb = np.sqrt(np.mean(a * a)), np.sqrt(np.mean(b * b))
    if min(ra, rb) < max(10 ** (-50 / 20), body / 4) or abs(_db(ra) - _db(rb)) > 3:
        return 0
    window = np.hanning(n)
    fa, fb = np.abs(np.fft.rfft(a * window)), np.abs(np.fft.rfft(b * window))
    edges = np.unique(np.geomspace(1, len(fa) - 1, 25).astype(int))
    ba = np.log1p(np.add.reduceat(fa, edges) * 1000)
    bb = np.log1p(np.add.reduceat(fb, edges) * 1000)
    if ba @ bb / (np.linalg.norm(ba) * np.linalg.norm(bb) + 1e-20) < 0.95:
        return 0
    step = np.median(np.abs(np.diff(np.concatenate([a, b]))))
    return int(abs(a[0] - b[-1]) <= max(4 * step, 1e-4))


def clipped_runs(y):
    runs = 0
    for ch in range(y.shape[1]):
        hot = np.abs(y[:, ch]) >= CLIP_LEVEL
        if hot.sum() < CLIP_RUN:
            continue
        edges = np.diff(np.concatenate([[0], hot.view(np.int8), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        runs += int(((ends - starts) >= CLIP_RUN).sum())
    return runs


def measure(y, sr, duration=None, tail=None):
    """All measures of decoded samples `y` (frames x channels). `tail` is the file's last samples when `y` is only
    its head (files over MAX_DECODE_S). Values that do not apply are None."""
    y = y if y.ndim == 2 else y[:, None]
    duration = len(y) / sr if duration is None else duration
    mono = y.mean(axis=1)
    sample_peak = float(np.abs(y).max()) if len(y) else 0.0
    out = dict.fromkeys(FIELDS[3:-1])
    out.update(duration_s=round(duration, 4), channels=y.shape[1], sample_rate=sr, analysed_s=round(len(y) / sr, 3),
               sample_peak_dbfs=_round(_db(sample_peak), 2), clipped_runs=clipped_runs(y))
    points = trim_points(mono, sample_peak)
    if points is None:  # digital silence or below the floor: only the meters apply
        out.update(lead_silence_s=round(duration, 4), tail_silence_s=0.0, sound_s=0.0, onsets=0, loop_ready=0)
        return out
    lead, last = points
    tail_mono = tail.mean(axis=1) if tail is not None else None
    if tail_mono is not None:
        end_points = trim_points(tail_mono, max(sample_peak, float(np.abs(tail).max())))
        tail_silence = len(tail_mono) / sr if end_points is None else (len(tail_mono) - 1 - end_points[1]) / sr
    else:
        tail_silence = (len(mono) - 1 - last) / sr
    sound = mono[lead:last + 1]
    detail = sound[:int(DETAIL_S * sr)]
    cum = loudness_curve(y, sr)
    lufs = integrated(cum, sr)
    rms = float(np.sqrt(np.mean(detail.astype(np.float64) ** 2)))
    attack, decay = attack_decay(detail, sr)
    spec = spectrum(detail, sr)
    out.update(lead_silence_s=round(lead / sr, 4), tail_silence_s=round(tail_silence, 4),
               sound_s=round(max(0.0, duration - lead / sr - tail_silence), 4),
               true_peak_dbtp=_round(true_peak(y), 2), crest_db=_round(_db(float(np.abs(detail).max())) - _db(rms), 2),
               lufs_integrated=_round(lufs, 2), lufs_short_max=_round(window_max(cum, sr, 3.0), 2),
               lufs_momentary_max=_round(window_max(cum, sr, 0.4), 2),
               attack_ms=_round(attack, 2), decay_ms=_round(decay, 1), onsets=onset_count(detail, sr),
               loop_ready=loop_ready(tail_mono if tail_mono is not None else mono, sr, head=mono))
    if spec:
        out.update(centroid_hz=round(spec[0], 1), rolloff_hz=round(spec[1], 1), flatness=round(spec[2], 5),
                   low_share=round(spec[3], 4))
        if spec[2] <= PITCH_MAX_FLATNESS and len(sound) >= 0.06 * sr:
            out['pitch_hz'] = _round(pitch(sound, sr), 2)
    return out


def _round(value, digits):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def describe_file(path):
    import soundfile as sf
    head_y, sr = load_audio(path, seconds=MAX_DECODE_S)
    if not len(head_y):
        raise ValueError('empty audio')
    try:
        duration = sf.info(path).duration
    except Exception:
        duration = None
    tail = None
    if duration and duration > MAX_DECODE_S + 1:
        tail, _ = load_audio(path, seconds=2.0, start=duration - 2.0)
    return measure(head_y, sr, duration if tail is not None else None, tail)


def _work(item):
    path, rel, size, mtime = item
    try:
        return rel, size, mtime, describe_file(path), None
    except Exception as exc:
        return rel, size, mtime, None, str(exc).strip() or type(exc).__name__


def library(root, only=None, out=None, workers=4):
    """Resumable pass. Rows whose size, mtime and describer version match are kept; readers keep the last row per path."""
    from concurrent.futures import ProcessPoolExecutor
    root = Path(root)
    out = Path(out) if out else root / 'AUDIO_DESCRIPTORS.csv'
    done = {}
    if out.exists():
        with out.open(encoding='utf-8', newline='') as f:
            done = {r['relative_path']: r for r in csv.DictReader(f)}
    files = sorted(p for p in root.rglob('*') if p.suffix.lower() in AUDIO and p.is_file()
                   and fnmatch.fnmatch(p.relative_to(root).as_posix().lower(), (only or '*').lower()))
    todo = []
    for p in files:
        rel, stat = p.relative_to(root).as_posix(), p.stat()
        old = done.get(rel)
        if not (old and old['size_bytes'] == str(stat.st_size) and old['mtime_ns'] == str(stat.st_mtime_ns)
                and old.get('describer') == VERSION):
            todo.append((p, rel, stat.st_size, stat.st_mtime_ns))
    print(f'{len(files)} files, {len(todo)} to describe', file=sys.stderr, flush=True)
    workers = max(1, min(workers, len(todo) // 25 + 1))  # starting a worker costs ~2 s of imports: few files, few workers
    if workers > 1:  # one math thread per worker: N workers x all-core BLAS/FFT threads oversubscribe the CPU
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
            os.environ[key] = '1'
    new = not out.exists()
    if not new and out.stat().st_size:
        with out.open('rb') as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b'\n':  # a pass stopped mid-row: end that row, so the next one starts on its own line
                with out.open('a', encoding='utf-8') as g:
                    g.write('\n')
    described, skipped, start = 0, 0, time.perf_counter()
    with out.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, FIELDS)
        if new:
            writer.writeheader()
        pool = ProcessPoolExecutor(workers) if workers > 1 else None
        results = pool.map(_work, todo, chunksize=8) if pool else map(_work, todo)
        try:
            for n, (rel, size, mtime, row, error) in enumerate(results, 1):
                if error:
                    skipped += 1
                    print(f'SKIP {rel}: {error}', file=sys.stderr, flush=True)
                else:
                    writer.writerow({**row, 'relative_path': rel, 'size_bytes': size, 'mtime_ns': mtime, 'describer': VERSION})
                    described += 1
                if n % 200 == 0 or n == len(todo):
                    f.flush()
                    print(f'{n}/{len(todo)}', file=sys.stderr, flush=True)
        finally:
            if pool:
                pool.shutdown()
    elapsed = time.perf_counter() - start
    summary = {'described': described, 'unchanged': len(files) - len(todo), 'skipped': skipped,
               'seconds_per_file': round(elapsed / max(1, len(todo)), 4)}
    print('SUMMARY ' + ' '.join(f'{k}={v}' for k, v in summary.items()), file=sys.stderr, flush=True)
    return summary


# ---------------------------------------------------------------------------------------------------------

def check():
    """Every measure on a synthetic signal whose answer is known in advance."""
    sr = 48000
    t = np.arange(sr) / sr
    rng = np.random.default_rng(7)
    near = lambda a, b, tol: a is not None and abs(a - b) <= tol

    # Loudness: the coefficients BS.1770 tabulates at 48 kHz, and a 997 Hz full-scale sine reads -3.01 LUFS.
    (b1, a1), (b2, a2) = k_weighting(48000)
    assert np.allclose(b1, [1.53512485958697, -2.69169618940638, 1.19839281085285], atol=1e-8), b1
    assert np.allclose(a1, [1, -1.69065929318241, 0.73248077421585], atol=1e-8), a1
    assert np.allclose(a2, [1, -1.99004745483398, 0.99007225036621], atol=1e-8), a2
    sine = np.sin(2 * np.pi * 997 * np.arange(sr * 3) / sr)[:, None]
    m = measure(sine, sr)
    assert near(m['lufs_integrated'], -3.01, 0.02) and near(m['lufs_short_max'], -3.01, 0.02), m
    assert near(m['true_peak_dbtp'], 0.0, 0.05), m['true_peak_dbtp']
    assert measure(sine[:int(0.3 * sr)] * 0.5, sr)['lufs_integrated'] is None, 'under 400 ms the standard has no integrated value'
    # Gating: 3 s loud then 3 s 25 dB quieter; the relative gate drops the quiet blocks (ungated would read ~3 dB lower).
    loud = 0.5 * np.sin(2 * np.pi * 997 * np.arange(sr * 3) / sr)
    gated = measure(np.concatenate([loud, loud * 10 ** (-25 / 20)]), sr)['lufs_integrated']
    assert near(gated, measure(loud, sr)['lufs_integrated'], 0.3), gated
    try:  # and FFmpeg's ebur128 meter agrees on stereo pink-ish noise at 44.1 kHz
        from audio_map import loudness_samples
        from scipy.signal import lfilter
        stereo = lfilter([1], [1, -0.9], rng.standard_normal((44100 * 4, 2)), axis=0) * 0.05
        ffmpeg = loudness_samples(stereo.astype('float32'), 44100)['integrated_lufs']
        ours = measure(stereo, 44100)['lufs_integrated']
        assert abs(ours - ffmpeg) <= 0.1, (ours, ffmpeg)
    except FileNotFoundError:
        print('NOTE no FFmpeg: loudness checked against the tabulated coefficients only')

    # Pitch: a 440 Hz tone gives 440 Hz; white noise is flat, has no pitch, and is called noisy.
    tone = measure(0.5 * np.sin(2 * np.pi * 440 * t), sr)
    assert near(tone['pitch_hz'], 440, 3) and tone['flatness'] < 0.01, tone
    noise = measure(0.3 * rng.standard_normal(sr), sr)
    assert noise['flatness'] > 0.3 and noise['pitch_hz'] is None, noise
    from scipy.signal import butter, sosfilt
    band = measure(sosfilt(butter(2, [700, 1100], 'bandpass', fs=sr, output='sos'), rng.standard_normal(sr)), sr)
    assert band['pitch_hz'] is None, 'band-limited noise has a low flatness but no pitch: tonal means a pitch'

    # Attack: a 5 ms linear ramp into a held tone measures about 4 ms (10% to 90% of a linear ramp is 80% of it).
    ramp = 0.5 * np.sin(2 * np.pi * 1000 * t) * np.minimum(1, t / 0.005)
    assert near(measure(ramp, sr)['attack_ms'], 4.0, 0.6), measure(ramp, sr)['attack_ms']
    # ... also after 30 ms of a faint lead-in above the silence floor: attack is the rise, not the time since the start.
    lead_in = np.concatenate([0.004 * np.sin(2 * np.pi * 1000 * t[:int(0.03 * sr)]), ramp])
    assert near(measure(lead_in, sr)['attack_ms'], 4.0, 0.6), measure(lead_in, sr)['attack_ms']
    # Decay: an exponential that loses 20 dB in 300 ms.
    decaying = np.sin(2 * np.pi * 800 * t) * 10 ** (-20 * t / 0.3 / 20)
    assert near(measure(decaying, sr)['decay_ms'], 300, 25), measure(decaying, sr)['decay_ms']

    # Silence: 50 ms of leading and 120 ms of trailing digital silence are reported, and the sound length excludes them.
    burst = np.concatenate([np.zeros(int(0.05 * sr)), 0.5 * np.sin(2 * np.pi * 600 * t[:int(0.2 * sr)]), np.zeros(int(0.12 * sr))])
    m = measure(burst, sr)
    assert near(m['lead_silence_s'], 0.05, 0.001) and near(m['tail_silence_s'], 0.12, 0.002) and near(m['sound_s'], 0.2, 0.003), m

    # Clipping: a sine driven to twice full scale is flagged; the same sine at 0.9 is not.
    assert measure(np.clip(2 * np.sin(2 * np.pi * 220 * t), -1, 1), sr)['clipped_runs'] > 100
    assert measure(0.9 * np.sin(2 * np.pi * 220 * t), sr)['clipped_runs'] == 0

    # Loop: a whole number of cycles over a steady bed loops; the same cut mid-cycle, or fading out, does not.
    two = np.arange(2 * sr) / sr
    bed = 0.05 * rng.standard_normal(2 * sr)
    seamless = 0.5 * np.sin(2 * np.pi * 220 * two) + bed
    assert measure(seamless, sr)['loop_ready'] == 1
    assert measure(seamless[:int(1.9011 * sr)], sr)['loop_ready'] == 0, 'cut mid-cycle: the seam jumps'
    assert measure(seamless * np.linspace(1, 0.2, len(seamless)), sr)['loop_ready'] == 0, 'fading: the ends differ in level'
    assert measure(burst, sr)['loop_ready'] == 0, 'silent ends are not a loop'
    assert measure(seamless * np.hanning(len(seamless)), sr)['loop_ready'] == 0, 'fading out and in again pulses'

    # Onsets: one click is one event; five clicks 150 ms apart are five.
    click = np.zeros(sr)
    for k in range(5):
        click[int((0.1 + 0.15 * k) * sr):int((0.1 + 0.15 * k) * sr) + 240] = rng.standard_normal(240) * np.hanning(240)
    assert measure(click[:int(0.25 * sr)], sr)['onsets'] == 1 and measure(click, sr)['onsets'] == 5
    # One hit with a wobbling resonant tail is still one event (librosa's default picker counts two).
    t8 = np.arange(int(0.8 * sr)) / sr
    hit = np.concatenate([rng.standard_normal(480) * np.hanning(480), np.zeros(len(t8) - 480)])
    ring = 0.5 * np.sin(2 * np.pi * 600 * t8) * np.exp(-t8 / 0.25) * (1 - 0.5 * (0.5 + 0.5 * np.cos(2 * np.pi * 12 * t8)))
    assert measure(hit + ring, sr)['onsets'] == 1, measure(hit + ring, sr)['onsets']

    # Brightness and depth: high-passed noise is brighter than low-passed noise; a 60 Hz tone is deep.
    from scipy.signal import butter, sosfilt
    hi = sosfilt(butter(4, 4000, 'highpass', fs=sr, output='sos'), rng.standard_normal(sr))
    lo = sosfilt(butter(4, 400, 'lowpass', fs=sr, output='sos'), rng.standard_normal(sr))
    assert measure(hi, sr)['centroid_hz'] > 4 * measure(lo, sr)['centroid_hz']
    assert measure(0.5 * np.sin(2 * np.pi * 60 * t), sr)['low_share'] > 0.9

    # Words: every measure a word names exists, and the query splitter turns plain words into filters.
    assert all(m in FIELDS or m.endswith('_pct') for row in WORDS for m, _, _ in row[2])
    assert set(MEASURE_EXPLAIN) == set(NUMBERS) - {'centroid_hz'}, 'every number the browser shows has a sentence'
    assert split_query('short punchy bright click') == ('click', ['short', 'punchy', 'bright'])
    assert split_query('a seamless rain loop') == ('a rain loop', ['loopable'])
    table = {'sound_s': np.array([0.1, 2.0, np.nan]), 'attack_ms': np.array([2.0, 50.0, 1.0]),
             'crest_db': np.array([20.0, 5.0, 20.0])}
    assert word_mask('short', table).tolist() == [True, False, False], 'an unmeasured file never passes a filter'
    assert word_mask('punchy', table).tolist() == [True, False, True]

    # The whole file path: a written WAV with leading silence comes back with the same answers.
    import tempfile
    import soundfile as sf
    with tempfile.TemporaryDirectory() as tmp:
        sf.write(Path(tmp) / 'a.wav', burst.astype('float32'), sr)
        library(tmp, workers=1)
        rows = list(csv.DictReader((Path(tmp) / 'AUDIO_DESCRIPTORS.csv').open(encoding='utf-8')))
        assert len(rows) == 1 and abs(float(rows[0]['lead_silence_s']) - 0.05) < 0.001, rows
        assert library(tmp, workers=1)['unchanged'] == 1, 'an unchanged file is not measured again'
        sf.write(Path(tmp) / 'b.wav', burst.astype('float32'), sr)
        with (Path(tmp) / 'AUDIO_DESCRIPTORS.csv').open('a', encoding='utf-8') as f:
            f.write('b.wav,123,4')  # a pass interrupted mid-row
        library(tmp, workers=1)
        assert read_descriptors(tmp)['b.wav']['lead_silence_s'] == rows[0]['lead_silence_s'], 'the next row starts on its own line'
    print('PASS audio_describe check: loudness -3.01 LUFS, pitch 440 Hz, 4 ms attack, 300 ms decay, 50 ms lead silence, '
          'clipping, loop seam, onsets 1/5, brightness, words as filters')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('describe', help='measure every audio file under --root (resumable)')
    d.add_argument('--root', type=Path, required=True)
    d.add_argument('--only', help='relative-path glob under --root')
    d.add_argument('--out', type=Path, help='CSV (default <root>/AUDIO_DESCRIPTORS.csv)')
    d.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    s = sub.add_parser('show')
    s.add_argument('files', nargs='+', type=Path)
    sub.add_parser('check')
    a = parser.parse_args()
    if a.cmd == 'check':
        return check()
    if a.cmd == 'show':
        result = {}
        for p in a.files:
            m = describe_file(p)
            table = {k: np.array([_float(m.get(k))]) for k in FIELDS}
            table['centroid_pct'] = np.array([np.nan])  # a percentile needs a library
            result[str(p)] = {**m, 'words': words_for(table, 0)}
        return print(json.dumps(result, indent=1))
    if not a.root.is_dir():
        parser.error(f'Library folder does not exist: {a.root}')
    library(a.root, a.only, a.out, a.workers)


if __name__ == '__main__':
    main()
