"""Print a track's tempo, beats, loud and quiet sections and strongest hits as JSON.

    python audio_map.py track.mp3      # any audio or video file FFmpeg can decode
    python audio_map.py --selftest

Music mapping needs librosa; shared meters need numpy and soundfile.
FFmpeg uses FFMPEG, WinGet or PATH. These are
measurements, not hearing: see references/music-and-beats.md for how to read them.
"""

import argparse
import json
import subprocess
import sys
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=None)
def media_tool(name="ffmpeg"):
    """FFMPEG override, its sibling ffprobe, WinGet, then PATH. No installation."""
    override = os.environ.get("FFMPEG")
    if override:
        target = Path(override) if name == "ffmpeg" else Path(override).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if not target.is_file():
            raise FileNotFoundError(target)
        return str(target)
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"]) / "Microsoft/WinGet/Packages"
        matches = sorted(base.glob(f"Gyan.FFmpeg*/*/bin/{name}.exe"))
        if matches:
            return str(matches[-1])
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f"Set FFMPEG to the ffmpeg executable or put {name} on PATH")
    return found


def probe(path):
    result = subprocess.run([media_tool("ffprobe"), "-v", "error", "-show_streams",
                             "-show_format", "-of", "json", str(path)],
                            check=True, capture_output=True, encoding="utf-8", errors="replace")
    return json.loads(result.stdout)


def load_audio(path, seconds=None, sample_rate=None, mono=False, start=0):
    """Share decoding between music mapping and ambience measurements."""
    import soundfile as sf

    try:
        with sf.SoundFile(path) as source:
            sr = source.samplerate
            if sample_rate is not None and sample_rate != sr:
                raise ValueError("FFmpeg resampling needed")
            source.seek(min(int(start * sr), len(source)))
            y = source.read(frames=-1 if seconds is None else int(seconds * sr),
                            dtype="float32", always_2d=True)
    except (sf.LibsndfileError, ValueError):
        meta = next(s for s in probe(path)["streams"] if s["codec_type"] == "audio")
        sr = sample_rate or int(meta["sample_rate"])
        channels = 1 if mono else int(meta["channels"])
        cmd = [media_tool(), "-v", "error", "-threads", "1", "-ss", str(start), "-i", str(path), "-vn"]
        if seconds is not None:
            cmd += ["-t", str(seconds)]
        result = subprocess.run(cmd + ["-ac", str(channels), "-ar", str(sr), "-f", "f32le", "pipe:1"],
                                check=True, capture_output=True)
        y = np.frombuffer(result.stdout, dtype="<f4").reshape(-1, channels)
    return (y.mean(axis=1) if mono else y), sr


def loudness(path, seconds=None):
    """EBU R128 integrated loudness, range and oversampled true peak."""
    cmd = [media_tool(), "-hide_banner", "-nostats", "-threads", "1"]
    if seconds is not None:
        cmd += ["-t", str(seconds)]  # input limit; an output limit meters extra decoded samples
    cmd += ["-i", str(path), "-map", "0:a:0", "-vn"]
    filters = (f"atrim=duration={seconds}," if seconds is not None else "") + "ebur128=peak=true"
    return _r128(cmd + ["-af", filters, "-f", "null", "-"])


def loudness_samples(y, sr):
    """Meter already decoded PCM if FFmpeg cannot read a damaged container."""
    if y.ndim == 1:
        y = y[:, None]
    cmd = [media_tool(), "-hide_banner", "-nostats", "-threads", "1", "-f", "f32le",
           "-ar", str(sr), "-ac", str(y.shape[1]), "-i", "pipe:0", "-af", "ebur128=peak=true",
           "-f", "null", "-"]
    return _r128(cmd, y.astype("<f4").tobytes())


def _r128(cmd, data=None):
    result = subprocess.run(cmd, input=data, check=True, capture_output=True)
    summary = result.stderr.decode(errors="replace").rsplit("Summary:", 1)[-1]
    fields = {}
    for key, pattern in [("integrated_lufs", r"I:\s+([-+\w.]+) LUFS"),
                         ("loudness_range_lu", r"LRA:\s+([-+\w.]+) LU"),
                         ("true_peak_dbtp", r"Peak:\s+([-+\w.]+) dBFS")]:
        match = re.search(pattern, summary)
        if not match:
            raise ValueError(f"Missing {key} in ebur128 summary: {summary}")
        fields[key] = float(match[1])
    return fields


def band_shares(y, sr):
    """Sum channel powers (no stereo cancellation), five Hz bands, Hann windows."""
    if y.ndim == 1:
        y = y[:, None]
    size = 4096
    power = np.zeros(size // 2 + 1)
    window = np.hanning(size)[:, None]
    for start in range(0, len(y), size):
        chunk = y[start:start + size]
        if len(chunk) < size:
            chunk = np.pad(chunk, ((0, size - len(chunk)), (0, 0)))
        power += (abs(np.fft.rfft(chunk * window, axis=0)) ** 2).sum(axis=1)
    freq = np.fft.rfftfreq(size, 1 / sr)
    bounds = (0, 300, 1500, 3500, 8000, float("inf"))
    total = power.sum()
    return [float(power[(freq >= a) & (freq < b)].sum() / total) if total else 0.0
            for a, b in zip(bounds, bounds[1:])]


def rms_blocks(y, sr, seconds):
    if y.ndim == 1:
        y = y[:, None]
    size = max(1, round(sr * seconds))
    # Keep a last partial second; discard it only for short onset windows.
    return np.array([np.sqrt(np.mean(y[i:i + size].astype(np.float64) ** 2))
                     for i in range(0, len(y), size)])


def analyze(path, sections=None, hits=12):
    y, sr = load_audio(path, sample_rate=22050, mono=True)
    return analyze_samples(y, sr, sections, hits)


def analyze_samples(y, sr, sections=None, hits=12):
    import librosa

    duration = len(y) / sr
    # trim=False keeps the beat grid running through a quiet intro or outro.
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo = float(np.atleast_1d(tempo)[0])
    if len(beat_frames) > 2:  # the tracked beats are finer than the tracker's tempo bins
        first, last = librosa.frames_to_time(beat_frames[[0, -1]], sr=sr)
        tempo = 60 * (len(beat_frames) - 1) / float(last - first)
    rms = librosa.feature.rms(y=y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    frames = mfcc.shape[1]

    def seconds(frame_indices):
        return [round(float(t), 2) for t in librosa.frames_to_time(frame_indices, sr=sr)]

    # ponytail: section count is duration/8 clamped to 3..12, which over-splits rather than
    # hides a change; pass --sections when the track's structure is known.
    k = sections or int(np.clip(round(duration / 8), 3, 12))
    # Boundaries land on beats when there are enough of them, else on a half-second grid.
    grid = beat_frames if len(beat_frames) >= 2 * k else np.arange(0, frames, sr // 1024)
    grid = librosa.util.fix_frames(grid, x_min=0, x_max=frames)
    synced = librosa.util.sync(mfcc, grid)
    starts = grid[librosa.segment.agglomerative(synced, min(k, synced.shape[1]))]
    ends = np.append(starts[1:], frames)
    means = np.array([rms[a:b].mean() for a, b in zip(starts, ends)])
    levels = 20 * np.log10(np.maximum(means, 1e-10) / max(means.max(), 1e-10))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    strongest = np.sort(onsets[np.argsort(onset_env[onsets])[::-1][:hits]])

    per_second = librosa.util.sync(rms, np.arange(0, frames, sr // 512), aggregate=np.mean)
    return {
        "duration": round(duration, 2),
        "tempo_bpm": round(tempo, 1),
        "beats": seconds(beat_frames),
        # level_db: 0 is the loudest section, more negative is quieter.
        "sections": [{"start": s, "end": e, "level_db": round(float(level), 1)}
                     for s, e, level in zip(seconds(starts), seconds(ends), levels)],
        "strongest_hits": seconds(strongest),
        # One value per second, 0 is the loudest second.
        "energy_db_per_second": [int(v) for v in np.round(
            librosa.amplitude_to_db(per_second, ref=np.max))],
    }


def selftest():
    import librosa

    # 120 BPM clicks, quiet for ten seconds then loud for ten.
    sr = 22050
    y = librosa.clicks(times=np.arange(0, 20, 0.5), sr=sr, length=20 * sr)
    y[:10 * sr] *= 0.1
    result = analyze_samples(y, sr, sections=3)
    quiet, loud, _tail = result["sections"]
    assert abs(result["tempo_bpm"] - 120) < 3, result["tempo_bpm"]
    assert abs(loud["start"] - 10) < 1, loud
    assert quiet["level_db"] < -15 and loud["level_db"] == 0, result["sections"]
    assert len(result["beats"]) >= 36, len(result["beats"])
    print("selftest passed")


def main():
    parser = argparse.ArgumentParser(
        description='Print tempo, beats, loud and quiet sections, and strongest hits as JSON.',
        epilog='Example: python scripts/audio_map.py track.wav',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file", nargs="?", help="audio or video file")
    parser.add_argument("--sections", type=int, help="number of sections to split into")
    parser.add_argument("--hits", type=int, default=12, help="how many strongest hits to list")
    parser.add_argument("--selftest", action="store_true", help='run synthetic checks without a library')
    args = parser.parse_args()
    if args.selftest or args.file == "check":
        return selftest()
    if not args.file:
        parser.error("give a file or --selftest")
    json.dump(analyze(args.file, args.sections, args.hits), sys.stdout)
    print()


if __name__ == "__main__":
    main()
