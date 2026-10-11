"""Index non-music from a library ledger. Names are guesses; listening is pending."""

import argparse
import csv
import fnmatch
import json
import math
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from audio_map import band_shares, load_audio, loudness, loudness_samples, probe, rms_blocks

BAND_FIELDS = ['band_0_300', 'band_300_1500', 'band_1500_3500', 'band_3500_8000', 'band_8000_plus']
FIELDS = ['relative_path', 'size_bytes', 'mtime_ns', 'duration_seconds', 'channels',
          'measured_seconds', 'meter_input', 'integrated_lufs', 'loudness_range_lu', 'true_peak_dbtp',
          *BAND_FIELDS, 'event_density_per_min', 'steadiness_cv', 'loop_hint',
          'biome_guess', 'source_guess', 'role_guess']


def texture(y, sr):
    # Remove slow/DC drift within each onset window; keep original energy for bands/CV.
    size = max(1, round(sr * 0.02))
    blocks = y[:len(y) // size * size].reshape(-1, size, y.shape[1])
    if not len(blocks):  # too short for an onset window
        return 0.0, 0.0
    levels = np.sqrt(np.mean((blocks - blocks.mean(axis=1, keepdims=True)) ** 2, axis=(1, 2)))
    floor = float(np.percentile(levels, 20))
    median = float(np.median(levels))
    mad = float(np.median(abs(levels - median)))
    threshold = max(floor * 2.5, median + 10 * mad, 1e-6)
    events, armed, last = 0, True, -1.0
    for i, value in enumerate(levels):
        time = i * 0.02
        if value < threshold * 0.8:
            armed = True
        if armed and value > threshold and time - last >= 0.18:
            events, armed, last = events + 1, False, time
    seconds = len(y) / sr
    one_second = rms_blocks(y, sr, 1)
    cv = float(one_second.std() / one_second.mean()) if one_second.mean() else 0.0
    return events * 60 / seconds, cv


def guesses(relative, duration, density):
    text = relative.lower()
    # Names alone, first specific match wins. A pack name is provenance, not a licence.
    rules = [('jungle', ('jungle', 'rainforest', 'tropical')),
             ('wetland/pond', ('wetland', 'pond', 'swamp', 'frog', 'marsh')),
             ('night', ('night', 'nocturn', 'owl')),
             ('forest', ('forest', 'woodland', 'woods')),
             ('meadow/birds', ('meadow', 'bird', 'aviary', 'songbird')),
             ('water', ('water', 'river', 'ocean', 'stream', 'rain', 'wave', 'sea')),
             ('wind', ('wind', 'breeze')),
             ('insects', ('insect', 'cricket', 'cicada', 'bug', 'mosquito'))]
    biome = next((name for name, words in rules if any(w in text for w in words)), 'unknown')
    parts = Path(relative).parts
    source = '/'.join(parts[:2]) if parts[0].lower() == 'audiohero' and len(parts) > 1 else parts[0]
    role = 'single call' if duration <= 8 and density > 0 else 'bed' if duration >= 10 else 'other'
    return biome, source, role


def measure(path, root):
    import soundfile as sf

    try:
        info = sf.info(path)
        duration, channels = info.duration, info.channels
    except sf.LibsndfileError:
        meta = probe(path)
        stream = next(s for s in meta['streams'] if s['codec_type'] == 'audio')
        duration = float(stream.get('duration', meta['format'].get('duration', 0)))
        channels = int(stream['channels'])
    y, sr = load_audio(path, seconds=60)
    if not len(y):
        raise ValueError(f'Empty audio: {path}')
    density, cv = texture(y, sr)
    relative = path.relative_to(root).as_posix()
    stat = path.stat()
    meter_input = 'file'
    try:
        meters = loudness(path, 60)
    except subprocess.CalledProcessError:
        meters = loudness_samples(y, sr)
        meter_input = 'decoded_pcm'
        if len(y) < 60 * sr:
            duration = len(y) / sr  # damaged headers can overstate the decodable length
        print(f'NOTE ebur128 used decoded PCM: {relative}', file=sys.stderr, flush=True)
    first = y[:round(sr * 0.5)]
    if meter_input == 'decoded_pcm' and len(y) < 60 * sr:
        last, tail_sr = y[-round(sr * 0.5):], sr
    else:
        last, tail_sr = load_audio(path, seconds=0.5, start=max(0, duration - 0.5))
    level_delta = abs(20 * np.log10(max(float(np.sqrt(np.mean(first ** 2))), 1e-10) /
                                  max(float(np.sqrt(np.mean(last ** 2))), 1e-10)))
    spectral_delta = sum(abs(a - b) for a, b in zip(band_shares(first, sr), band_shares(last, tail_sr)))
    biome, source, role = guesses(relative, duration, density)
    return dict(zip(FIELDS, [relative, stat.st_size, stat.st_mtime_ns, round(duration, 4), channels,
                            round(len(y) / sr, 4), meter_input, *meters.values(),
                            *[round(v, 6) for v in band_shares(y, sr)], round(density, 3), round(cv, 5),
                            'possible' if duration >= 1 and level_delta <= 3 and spectral_delta <= 0.2 else 'unlikely',
                            biome, source, role]))


def selected(root, only):
    if only:
        return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in
                      {'.wav', '.mp3', '.ogg', '.flac', '.aif', '.aiff', '.m4a', '.aac', '.opus'}
                      and fnmatch.fnmatch(p.relative_to(root).as_posix().lower(), only.lower()))
    ledger = root / 'AUDIO_LEDGER_files.csv'
    with ledger.open(encoding='utf-8-sig', newline='') as source:
        rows = list(csv.DictReader(source))
    if not rows or not {'relative_path', 'kind_guess'} <= rows[0].keys():
        raise ValueError('Ledger needs relative_path and kind_guess; use --only for other libraries')
    # Unknown is not evidence of non-music. The ledger explicitly marks sfx and voice.
    return sorted({root / r['relative_path'] for r in rows if r['kind_guess'].lower() in {'sfx', 'voice'}})


def write_guide(out, rows):
    nature = [r for r in rows if r['biome_guess'] != 'unknown' and r['role_guess'] in {'bed', 'single call'}]
    groups = sorted({(r['biome_guess'], r['source_guess']) for r in nature})
    lines = ['# AUDIO AMBIENCE INDEX', '', 'Paths use forward slashes relative to the library root, as in AUDIO_LEDGER.md.',
             f'{len(rows):,} measured non-music files. Biome, source and role are name/duration guesses. Listening pending.',
             'Read the ledger licence column before use. Measurements do not establish rights or animal identity.', '',
             'LUFS/LRA/true peak, energy shares and density cover the first 60 s (or the shorter whole file).',
             'Density counts 20 ms rises above the file floor; impacts/voice also score. It is not an animal detector.',
             'meter_input=decoded_pcm flags containers FFmpeg rejected; soundfile decoded them for the same EBU filter.',
             'CV is the coefficient of variation of 1 s linear RMS. Lower means steadier.',
             'Loop hint: actual first/last 0.5 s within 3 dB and band-share distance <= 0.2; audition the seam.',
             'Bed ~30 dB below voice; bright (>3.5 kHz) beds lower still. One biome/source per scene.', '',
             '## Grouped beds and single calls (guesses)', '',
             'One measured candidate per role/group; highest density first. Counts include every matching file.',
             'Each row pairs the same biome/source when both roles exist. Missing matches remain unpaired.',
             'Candidate numbers: seconds / LUFS / LRA LU / dBTP / onsets per min / CV / bright share / loop hint.', '',
             '| Biome guess / source pack guess | Beds: count + candidate | Single calls: count + candidate |',
             '|---|---|---|']
    def entry(choices):
        if not choices:
            return '0; no measured match'
        r = sorted(choices, key=lambda r: (-float(r['event_density_per_min']), float(r['steadiness_cv'])))[0]
        bright = float(r['band_3500_8000']) + float(r['band_8000_plus'])
        return (f"{len(choices)}; `{r['relative_path']}`; {float(r['duration_seconds']):.2f}s / "
                f"{float(r['integrated_lufs']):.1f} / {float(r['loudness_range_lu']):.1f} / "
                f"{float(r['true_peak_dbtp']):.1f} / {float(r['event_density_per_min']):.1f} / "
                f"{float(r['steadiness_cv']):.2f} / {bright:.0%} / {r['loop_hint']}").replace('|', r'\|')
    for biome, source in groups[:96]:
        group = [r for r in nature if (r['biome_guess'], r['source_guess']) == (biome, source)]
        beds = [r for r in group if r['role_guess'] == 'bed']
        calls = [r for r in group if r['role_guess'] == 'single call']
        lines += [f"| {biome} / {source} | {entry(beds)} | {entry(calls)} |"]
    if len(groups) > 96:
        lines += ['', 'More groups are in the CSV; filter biome_guess, source_guess and role_guess.']
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def index(root, only=None, out=None):
    root = Path(root).resolve()
    out = Path(out or root / 'AUDIO_AMBIENCE_INDEX.csv').resolve()
    paths = selected(root, only)
    if not paths:
        raise ValueError('No selected audio files')
    old = {}
    if out.exists():
        with out.open(encoding='utf-8-sig', newline='') as source:
            reader = csv.DictReader(source)
            legacy = [k for k in FIELDS if k != 'meter_input']
            if reader.fieldnames not in (FIELDS, legacy):
                raise ValueError('Existing CSV has a different schema; choose a new --out')
            old = {}
            for row in reader:
                row.setdefault('meter_input', 'file')
                if all(row.get(k) is not None for k in FIELDS):
                    old[row['relative_path']] = row
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, pending, skipped, errors = [], [], 0, []
    for p in paths:
        if not p.resolve().is_relative_to(root):
            raise ValueError(f'Ledger path escapes root: {p}')
        key, stat = p.relative_to(root).as_posix(), p.stat()
        row = old.get(key)
        if row and row['size_bytes'] == str(stat.st_size) and row['mtime_ns'] == str(stat.st_mtime_ns):
            rows.append(row)
            skipped += 1
        else:
            pending.append(p)
    # Appending each complete row preserves progress after interruption. No detached jobs.
    with out.open('w', encoding='utf-8', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        target.flush()
        def attempt(p):
            try:
                return measure(p, root), None
            except Exception as exc:
                return None, f'{p.relative_to(root)}: {exc}'
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            for count, (row, error) in enumerate(pool.map(attempt, pending), 1):
                if error:
                    errors.append(error)
                    print('ERROR ' + error, file=sys.stderr, flush=True)
                else:
                    writer.writerow(row)
                    rows.append(row)
                target.flush()
                if count % 100 == 0:
                    print(f'Indexed {count}/{len(pending)} new; {skipped} unchanged; {len(errors)} errors', flush=True)
    write_guide(out, rows)
    result = {'selected': len(paths), 'measured': len(rows) - skipped, 'skipped': skipped, 'errors': len(errors)}
    print(json.dumps(result), flush=True)
    if errors:
        raise ValueError(f'{len(errors)} files failed; saved successes, rerun to retry')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    cmd = commands.add_parser('index')
    cmd.add_argument('root', type=Path)
    cmd.add_argument('--only', help='relative-path glob; bypass ledger classification')
    cmd.add_argument('--out', type=Path)
    commands.add_parser('show').add_argument('file', type=Path)
    commands.add_parser('check')
    args = parser.parse_args()
    if args.command == 'check':
        check()
    elif args.command == 'show':
        file = args.file.resolve()
        root = next((p for p in file.parents if (p / 'AUDIO_LEDGER_files.csv').is_file()), file.parent.parent)
        row = measure(file, root)
        # Silence can have a -inf peak; keep valid JSON and label it explicitly.
        row = {k: str(v) if isinstance(v, float) and not math.isfinite(v) else v for k, v in row.items()}
        print(json.dumps(row, indent=2, allow_nan=False))
    else:
        index(args.root, args.only, args.out)


def check():
    import tempfile
    from unittest.mock import patch
    import soundfile as sf

    sr, seconds = 24000, 12
    assert texture(np.zeros((10, 1)), sr) == (0.0, 0.0)
    rng = np.random.default_rng(7)
    white = rng.normal(size=sr * seconds)
    spectrum = np.fft.rfft(white)
    spectrum /= np.sqrt(np.maximum(np.fft.rfftfreq(len(white), 1 / sr), 1))
    bed = np.fft.irfft(spectrum, n=len(white))
    bed *= 0.015 / np.std(bed)
    calls = bed.copy()
    for start in np.arange(1, seconds, 2):
        t = np.arange(int(0.25 * sr)) / sr
        chirp = 0.35 * np.sin(2 * np.pi * (4500 * t + 1500 * t * t)) * np.hanning(len(t))
        calls[int(start * sr):int(start * sr) + len(t)] += chirp
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
        root = Path(tmp)
        sf.write(root / "forest-bed.wav", bed, sr)
        sf.write(root / "forest-calls.wav", calls, sr)
        a, b = (measure(p, root) for p in (root / "forest-bed.wav", root / "forest-calls.wav"))
        assert a["event_density_per_min"] < 5, a
        assert b["event_density_per_min"] >= 25, b
        assert b["integrated_lufs"] > a["integrated_lufs"] + 10, (a, b)
        assert -70 < b["true_peak_dbtp"] < 0, b
        assert b["steadiness_cv"] > a["steadiness_cv"] + 0.25, (a, b)
        assert b["band_3500_8000"] > a["band_3500_8000"] + 0.25, (a, b)
        assert abs(sum(b[k] for k in BAND_FIELDS) - 1) < 0.001, b
        # A loud transient just after the minute must not enter any R128 measurement.
        t = np.arange(sr * 61) / sr
        late = 0.01 * np.sin(2 * np.pi * 1000 * t)
        late[int(sr * 60.01):] *= 90
        sf.write(root / 'late.wav', late, sr)
        sf.write(root / 'minute.wav', late[:60 * sr], sr)
        minute = loudness(root / 'minute.wav')
        assert loudness(root / 'late.wav', 60) == minute, minute
        assert abs(minute['integrated_lufs'] + 43) <= 0.2, minute
        pcm = loudness_samples(late[:60 * sr], sr)
        assert all(abs(pcm[k] - minute[k]) <= 0.1 for k in minute), (pcm, minute)
        (root / 'late.wav').unlink()
        (root / 'minute.wav').unlink()
        with patch(__name__ + '.loudness', side_effect=subprocess.CalledProcessError(1, 'fixture')):
            rescued = measure(root / 'forest-bed.wav', root)
        assert rescued['meter_input'] == 'decoded_pcm', rescued
        assert all(abs(rescued[k] - a[k]) <= 0.1 for k in
                   ('integrated_lufs', 'loudness_range_lu', 'true_peak_dbtp')), (rescued, a)
        out = root / "index.csv"
        result = index(root, "*.wav", out)
        assert result["measured"] == 2, result
        with patch(__name__ + ".measure", side_effect=AssertionError("resume re-measured")):
            result = index(root, "*.wav", out)
        assert result["skipped"] == 2 and result["measured"] == 0, result
        sf.write(root / 'music.wav', bed, sr)
        sf.write(root / 'unknown.wav', bed, sr)
        (root / 'AUDIO_LEDGER_files.csv').write_text(
            'relative_path,kind_guess\nforest-bed.wav,sfx\nforest-calls.wav,voice\nmusic.wav,music\nunknown.wav,unknown\n')
        assert {p.name for p in selected(root, None)} == {'forest-bed.wav', 'forest-calls.wav'}
        assert {p.name for p in selected(root, 'music*')} == {'music.wav'}
    print("PASS audio_index check: pink bed, chirps, bands, steadiness, resume")



if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"FAIL audio_index: {exc}", file=sys.stderr)
        sys.exit(1)
