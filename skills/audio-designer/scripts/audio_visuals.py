"""Cache whole-file peak envelopes, spectrograms and audition loudness beside AUDIO_DESCRIPTORS.csv.

The browser reads these small caches; decoding happens once per size/mtime/version change. Audio is read in
bounded blocks, so a long recording does not need a whole-file PCM array in memory. No models are used.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import struct
import zlib

import numpy as np

VERSION = 'v3'  # v2: streaming true-peak context; v3: PNG needs only numpy and stdlib
BINS = 128
CACHE = 'AUDIO_VISUALS'


def _source(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(str(path))
    return root, path


def cache_path(root, path):
    root, path = _source(root, path)
    folder = root / CACHE
    if not folder.resolve().is_relative_to(root):
        raise ValueError('The visual cache must stay inside the library')
    name = hashlib.sha256(path.relative_to(root).as_posix().encode('utf-8')).hexdigest()
    return folder / (name + '.json')


def _stamp(path):
    stat = path.stat()
    return {'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'version': VERSION,
            'source_stamp': f'{VERSION}:{stat.st_size}:{stat.st_mtime_ns}'}


def _atomic(path, body):
    path.parent.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as target:
        temporary = Path(target.name)
        target.write(body)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _blocks(path):
    """Yield (sample rate, frame count, block iterator); soundfile first, installed FFmpeg otherwise."""
    import soundfile as sf
    try:
        source = sf.SoundFile(path)
    except (sf.LibsndfileError, ValueError):
        source = None
    if source is not None:
        try:
            yield source.samplerate, len(source), source.blocks(blocksize=65536, dtype='float32', always_2d=True)
        finally:
            source.close()
        return
    from audio_map import media_tool, probe
    meta = probe(path)
    stream = next((s for s in meta['streams'] if s['codec_type'] == 'audio'), None)
    if stream is None:
        raise ValueError('The file has no audio stream')
    sr, channels = int(stream['sample_rate']), int(stream['channels'])
    duration = float(stream.get('duration') or meta.get('format', {}).get('duration') or 0)
    if duration <= 0:
        raise ValueError('Cannot determine audio duration')
    command = [media_tool(), '-v', 'error', '-threads', '1', '-i', str(path), '-vn', '-ac', str(channels),
               '-ar', str(sr), '-f', 'f32le', 'pipe:1']
    with tempfile.TemporaryFile() as errors, subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors) as process:
        def decoded():
            while True:
                raw = process.stdout.read(65536 * channels * 4)
                if not raw:
                    break
                yield np.frombuffer(raw, dtype='<f4').reshape(-1, channels)
            if process.wait():
                errors.seek(0)
                raise ValueError(errors.read().decode(errors='replace')[-300:])
        try:
            yield sr, max(1, round(duration * sr)), decoded()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def _analyse(path):
    """Whole-file channel-safe peaks and maximum K-weighted 3 s loudness; bounded PCM memory."""
    from scipy.signal import lfilter
    import audio_describe as describe
    peaks, sample_peak, true_peak = np.zeros(BINS), 0., -np.inf
    powers, pending, event, position = [], np.empty(0), [], 0
    with _blocks(path) as (sr, frames, blocks):
        filters = describe.k_weighting(sr)
        states = None
        peak_tail = None
        hop = max(1, round(sr * .1))
        for y in blocks:
            if not np.isfinite(y).all():
                raise ValueError('Non-finite audio samples')
            if not len(y):
                continue
            level = np.max(np.abs(y), axis=1)
            sample_peak = max(sample_peak, float(level.max()))
            # Delay the last 64 samples until the next block supplies the filter's right context.
            # Retain another 64 on the left; only the central region contributes to the meter.
            # This avoids both missing a peak across a block join and inventing peaks at padded joins.
            peak_buffer = y if peak_tail is None else np.concatenate([peak_tail, y])
            first = 0 if peak_tail is None else max(0, len(peak_tail) - 64)
            last = max(first, len(peak_buffer) - 64)
            if last > first:
                true_peak = max(true_peak, describe.true_peak(peak_buffer, start=first, end=last))
            peak_tail = peak_buffer[-128:].copy()
            # Bin by absolute frame position rather than block boundaries: every source sample contributes.
            bins = np.minimum(BINS - 1, (np.arange(position, position + len(y), dtype=np.int64) * BINS // frames))
            np.maximum.at(peaks, bins, level)
            if frames < 3 * sr:
                event.append(y.copy())
            if states is None:
                states = [[np.zeros(2) for _ in filters] for _ in range(y.shape[1])]
            energy = np.zeros(len(y))
            for ch in range(y.shape[1]):
                x = y[:, ch].astype(np.float64)
                for k, (b, a) in enumerate(filters):
                    x, states[ch][k] = lfilter(b, a, x, zi=states[ch][k])
                energy += x * x
            energy = np.concatenate([pending, energy])
            complete = len(energy) // hop * hop
            powers.extend(energy[:complete].reshape(-1, hop).sum(axis=1).tolist())
            pending = energy[complete:]
            position += len(y)
        if peak_tail is not None:
            true_peak = max(true_peak, describe.true_peak(peak_tail, start=max(0, len(peak_tail) - 64)))
    if not position:
        raise ValueError('empty audio')
    duration = position / sr
    loudness, basis = None, 'silence'
    if sample_peak > 1e-9:
        if duration >= 3 and len(powers) >= 30:
            cumulative = np.concatenate([[0.], np.cumsum(powers)])
            energy = float(np.max(cumulative[30:] - cumulative[:-30])) / (hop * 30)
            loudness, basis = float(describe._lufs(energy)), 'short-term 3 s'
        elif event:
            y = np.concatenate(event)
            # Trim only this short-event fallback; keep channel power, including opposed stereo.
            level = np.max(np.abs(y), axis=1)
            active = np.flatnonzero(level > max(1e-6, sample_peak * 10 ** (-50 / 20)))
            if len(active):
                y = y[active[0]:active[-1] + 1]
                curve = describe.loudness_curve(y, sr)
                loudness = float(describe._lufs(curve[-1] / len(y)))
                basis = 'event K-weighted (under 3 s)'
    if loudness is not None and not math.isfinite(loudness):
        loudness = None
    return {'duration_s': round(duration, 6), 'peaks': np.round(peaks, 6).tolist(),
            'sample_peak': round(sample_peak, 8), 'true_peak_dbtp': round(true_peak, 6) if math.isfinite(true_peak) else None,
            'match_lufs': round(loudness, 6) if loudness is not None else None, 'match_basis': basis}


def waveform(root, path):
    root, path = _source(root, path)
    cache = cache_path(root, path)
    stamp = _stamp(path)
    try:
        old = json.loads(cache.read_text(encoding='utf-8'))
        if all(old.get(k) == v for k, v in stamp.items()) and len(old.get('peaks', [])) == BINS:
            return old
    except (FileNotFoundError, ValueError, TypeError):
        pass
    result = _analyse(path)
    if _stamp(path) != stamp:
        raise ValueError('Audio changed while its waveform was being prepared; retry')
    result.update(stamp, path=path.relative_to(root).as_posix())
    _atomic(cache, json.dumps(result, separators=(',', ':'), allow_nan=False).encode('utf-8'))
    # An old spectrogram belongs to the old source stamp, even if its file remains in the folder.
    cache.with_suffix('.png.stamp').unlink(missing_ok=True)
    return result


def _spectrogram(path):
    """Whole-file overview: log frequency, channel-averaged power, fixed display size."""
    width, height = 700, 180
    power = np.zeros((height, width))
    with _blocks(path) as (sr, frames, blocks):
        size, hop, offset = 2048, 1024, 0
        window = np.hanning(size)
        frequency = np.geomspace(40, min(sr / 2, 20000), height)
        bins = np.minimum(size // 2, np.round(frequency * size / sr).astype(int))
        for y in blocks:
            if len(y) < size:
                y = np.pad(y, ((0, size - len(y)), (0, 0)))
            starts = np.arange(0, len(y) - size + 1, hop)
            chunks = np.stack([y[k:k + size] for k in starts])
            spectrum = np.mean(np.abs(np.fft.rfft(chunks * window[None, :, None], axis=1)) ** 2, axis=2)
            columns = np.minimum(width - 1, ((offset + starts + size // 2) * width // frames).astype(int))
            for column in np.unique(columns):
                power[:, column] = np.maximum(power[:, column], spectrum[columns == column][:, bins].max(axis=0))
            offset += min(len(y), frames - offset)
        # Fill unsampled display columns for short files, preserving all measured frequency bands.
        available = np.flatnonzero(power.max(axis=0) > 0)
        if len(available):
            for row in range(height):
                power[row] = np.interp(np.arange(width), available, power[row, available])
        relative_db = 10 * np.log10(np.maximum(power, 1e-20) / max(float(power.max()), 1e-20))
        value = np.clip((relative_db + 80) / 80, 0, 1)[::-1]
        # A restrained dark-to-teal-to-amber palette without a plotting dependency.
        stops = np.array([[9, 19, 33], [18, 62, 86], [35, 165, 170], [234, 203, 118]], dtype=float)
        colors = np.stack([np.interp(value, [0, .35, .7, 1], stops[:, ch]) for ch in range(3)], axis=-1).astype('uint8')
        image = np.full((224, 720, 3), (9, 19, 33), dtype=np.uint8)
        image[22:22 + height, 10:10 + width] = colors
        return _rgb_png(image)


def _rgb_png(rgb):
    """Encode a standard 8-bit RGB PNG with filter-0 scanlines; no optional imaging dependency."""
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError('PNG raster must contain RGB channels')
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    header = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    rows = b''.join(b'\0' + row.tobytes() for row in np.asarray(rgb, dtype=np.uint8))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(rows, 9)) + chunk(b'IEND', b'')


def spectrogram(root, path):
    root, path = _source(root, path)
    waveform(root, path)  # invalidate both visual products when the audio changes
    cache = cache_path(root, path)
    image, stampfile = cache.with_suffix('.png'), cache.with_suffix('.png.stamp')
    stamp = _stamp(path)
    try:
        if image.is_file() and json.loads(stampfile.read_text()) == stamp:
            return image
    except (FileNotFoundError, ValueError):
        pass
    body = _spectrogram(path)
    if _stamp(path) != stamp:
        raise ValueError('Audio changed while its spectrogram was being prepared; retry')
    _atomic(image, body)
    _atomic(stampfile, json.dumps(stamp).encode())
    return image


def match_gains(waves, target=-18., peak=-1.):
    """Use one shared loudness target with a peak ceiling; attenuate only, so HTML volume can apply the gains.

    Silence stays at unity. Keeping the loudness difference in dB before conversion is essential: amplitude
    gain is 10 ** (dB / 20), while 10 ** (dB / 10) would square the correction.
    """
    valid = [w for w in waves if w.get('match_lufs') is not None]
    if valid:
        target = min([target] + [w['match_lufs'] for w in valid]
                     + [w['match_lufs'] + peak - w['true_peak_dbtp'] for w in valid if w.get('true_peak_dbtp') is not None])
    gains = []
    for w in waves:
        db = target - w['match_lufs'] if w.get('match_lufs') is not None else 0.
        gains.append({'gain_db': round(db, 6), 'gain': 10 ** (db / 20)})
    return target, gains
