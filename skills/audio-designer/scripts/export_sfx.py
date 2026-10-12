"""Export chosen sounds to a game project: trimmed, normalised, converted, cleanly named, with a manifest.

    python export_sfx.py --root D:/MySounds --list selection.txt --out D:/MyGame/Assets/Audio/SFX
    python export_sfx.py --root D:/MySounds --list selection.txt --out OUT --format ogg --rate 44100
    python export_sfx.py --root D:/MySounds --list selection.txt --out OUT --lufs -18     # loudness for long files
    python export_sfx.py check

selection.txt holds one file per line, relative to --root (as the browser copies them) or absolute.
Each file: leading and trailing silence trimmed (PRE_ROLL_MS kept before the sound, a short fade after it),
resampled, normalised to a true-peak target (-1 dBTP by default) or, with --lufs, files of at least
--lufs-min-seconds to an integrated loudness (never pushing the true peak above the peak target), written as
16- or 24-bit WAV or OGG Vorbis, and named by --pattern from {category}, {name}, {nn}, {pack} and {stem}.
MANIFEST.csv in the output folder lists each source, its pack and any licence file found in the pack, every
processing step and the measured descriptors. Originals are only read; the output folder may not be inside
the library, so exported copies never end up tagged as library sounds.
"""

import argparse
import csv
import hashlib
import json
import re
from math import gcd
from pathlib import Path

import numpy as np

import audio_describe
from audio_map import load_audio
from library_metadata import relative_path, write_csv_atomic

PRE_ROLL_MS, FADE_OUT_MS = 5.0, 5.0
FORMATS = {'wav16': ('WAV', 'PCM_16', '.wav'), 'wav24': ('WAV', 'PCM_24', '.wav'), 'ogg': ('OGG', 'VORBIS', '.ogg')}
LICENCE = re.compile(r'licen[cs]e|eula|terms|readme|credits', re.I)
MANIFEST = ['output_file', 'source_path', 'pack', 'licence_files', 'steps', 'gain_db', 'output_true_peak_dbtp',
            'output_lufs_integrated', 'output_sound_s', 'output_format', 'output_rate'] + \
           [f'source_{f}' for f in audio_describe.FIELDS[3:-1]]


def slug(text):
    return re.sub(r'_+', '_', re.sub(r'[^a-z0-9]+', '_', text.lower())).strip('_') or 'sound'


def licence_files(root, rel):
    """Licence or readme files at the top of the source's pack (two levels), relative to the root."""
    parts = relative_path(rel).split('/')
    if len(parts) < 2:
        return []
    pack = Path(root) / parts[0]
    found = [p for p in list(pack.glob('*')) + list(pack.glob('*/*')) if p.is_file() and LICENCE.search(p.name)
             and p.suffix.lower() in {'.txt', '.pdf', '.md', '.rtf', '.html', '.htm', '.doc', '.docx'}]
    return sorted(relative_path(p.relative_to(root)) for p in found)


def best_tags(root):
    """Each file's studio tag with the most supporting sources, from AUDIO_VOCAB_TAGS.csv when present."""
    path, best = Path(root) / 'AUDIO_VOCAB_TAGS.csv', {}
    if path.exists():
        with path.open(encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                n = int(r.get('sources') or 0)
                if n >= 2 and n > best.get(r['relative_path'], ('', 0))[1]:
                    best[r['relative_path']] = (r['tag'], n)
    return {p: t for p, (t, _) in best.items()}


def process(y, sr, rate, mode, peak_db, lufs, lufs_min_seconds):
    """(samples, steps, gain) for one decoded file. y is frames x channels."""
    steps = []
    mono = y.mean(axis=1)
    points = audio_describe.trim_points(mono, float(np.abs(y).max()))
    if points:
        start = max(0, points[0] - int(PRE_ROLL_MS / 1000 * sr))
        end = min(len(y), points[1] + 1)
        if start or end < len(y):
            steps.append(f'trim {start / sr * 1000:.1f} ms lead (kept {PRE_ROLL_MS:g} ms pre-roll), {(len(y) - end) / sr * 1000:.1f} ms tail')
            cut_tail = end < len(y)
            y = y[start:end].copy()
            if cut_tail:  # the tail ends below the silence floor, not at zero: fade so the cut cannot click
                fade = min(len(y), int(FADE_OUT_MS / 1000 * sr))
                y[len(y) - fade:] *= np.linspace(1, 0, fade)[:, None]
                steps.append(f'fade out {FADE_OUT_MS:g} ms')
    if rate and rate != sr:
        from scipy.signal import resample_poly
        g = gcd(rate, sr)
        y = resample_poly(y, rate // g, sr // g, axis=0)
        steps.append(f'resample {sr} -> {rate} Hz')
        sr = rate
    tp = audio_describe.true_peak(y)
    gain = peak_db - tp
    seconds = len(y) / sr
    if mode == 'lufs' and seconds >= lufs_min_seconds:
        measured = audio_describe.integrated(audio_describe.loudness_curve(y, sr), sr)
        if measured is not None:
            gain = min(lufs - measured, peak_db - tp)
            steps.append(f'loudness {measured:.1f} -> {measured + gain:.1f} LUFS' +
                         (f' (held at the {peak_db:g} dBTP peak ceiling)' if gain < lufs - measured - 1e-9 else ''))
        else:
            steps.append(f'peak {tp:.2f} -> {peak_db:g} dBTP (too quiet to meter integrated loudness)')
    else:
        steps.append(f'peak {tp:.2f} -> {peak_db:g} dBTP')
    return y * 10 ** (gain / 20), sr, steps, gain


def export(root, paths, out, fmt='wav16', rate=48000, mode='peak', peak_db=-1.0, lufs=-18.0, lufs_min_seconds=3.0,
           pattern='{category}_{name}_{nn}', category=None):
    """Export `paths` (relative to root) to `out`. Returns the manifest rows."""
    import soundfile as sf
    from audio_search import clean_name
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or out.is_relative_to(root):
        raise ValueError('Choose an output folder outside the library: exported copies must not mix with the originals')
    if fmt not in FORMATS:
        raise ValueError(f'format must be one of {", ".join(FORMATS)}')
    out.mkdir(parents=True, exist_ok=True)
    container, subtype, suffix = FORMATS[fmt]
    tags, counters, rows = best_tags(root), {}, []
    for rel in paths:
        rel = relative_path(Path(rel).resolve().relative_to(root) if Path(rel).is_absolute() else rel)
        source = (root / rel).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise FileNotFoundError(rel)
        y, sr = load_audio(source)
        if not len(y):
            raise ValueError(f'{rel}: empty audio')
        described = audio_describe.measure(y, sr)
        y, out_sr, steps, gain = process(y, sr, rate, mode, peak_db, lufs, lufs_min_seconds)
        cat = slug(category or tags.get(rel) or 'sfx')
        pack = rel.split('/')[0] if '/' in rel else ''
        words = [w for w in clean_name(rel, pack).split() if w not in cat.split('_')]
        name = slug(' '.join(words)) if words else 'sound'
        key = (cat, name)
        counters[key] = counters.get(key, 0) + 1
        stem = slug(Path(rel).stem)
        filename = pattern.format(category=cat, name=name, nn=f'{counters[key]:02d}', pack=slug(pack), stem=stem)
        target = out / (slug(filename) + suffix)
        while target.exists():  # never overwrite an earlier export
            counters[key] += 1
            target = out / (slug(pattern.format(category=cat, name=name, nn=f'{counters[key]:02d}', pack=slug(pack), stem=stem)) + suffix)
        sf.write(target, np.clip(y, -1, 1).astype('float32'), out_sr, subtype=subtype, format=container)
        steps.append(f'write {fmt} {out_sr} Hz')
        written, _ = load_audio(target)
        check = audio_describe.measure(written, out_sr)
        rows.append({'output_file': target.name, 'source_path': rel, 'pack': pack, 'licence_files': ';'.join(licence_files(root, rel)),
                     'steps': '; '.join(steps), 'gain_db': round(gain, 2), 'output_true_peak_dbtp': check['true_peak_dbtp'],
                     'output_lufs_integrated': check['lufs_integrated'], 'output_sound_s': check['sound_s'],
                     'output_format': fmt, 'output_rate': out_sr,
                     **{f'source_{k}': v for k, v in described.items() if f'source_{k}' in MANIFEST}})
    manifest = out / 'MANIFEST.csv'
    old = []
    if manifest.exists():
        with manifest.open(encoding='utf-8', newline='') as f:
            old = list(csv.DictReader(f))
    write_csv_atomic(manifest, MANIFEST, old + rows)
    return rows


def read_list(path):
    return [line.strip().strip('"') for line in Path(path).read_text(encoding='utf-8-sig').splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


def check():
    import tempfile
    import soundfile as sf
    sr = 44100
    t = np.arange(int(0.3 * sr)) / sr
    click = np.concatenate([np.zeros(int(0.2 * sr)), 0.25 * np.sin(2 * np.pi * 900 * t) * np.exp(-t / 0.05), np.zeros(int(0.3 * sr))])
    long = 0.05 * np.random.default_rng(1).standard_normal(sr * 5)
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp) / 'lib', Path(tmp) / 'game'
        (root / 'Pack' / 'UI').mkdir(parents=True)
        sf.write(root / 'Pack/UI/UI_Click_01.wav', np.stack([click, click], 1).astype('float32'), sr)
        sf.write(root / 'Pack/Wind Loop.wav', long.astype('float32'), sr)
        (root / 'Pack' / 'License.txt').write_text('Royalty free.', encoding='utf-8')
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
        (root / 'sel.txt').write_text('Pack/UI/UI_Click_01.wav\n' + str(root / 'Pack/Wind Loop.wav') + '\n', encoding='utf-8')
        rows = export(root, read_list(root / 'sel.txt'), out, fmt='wav16', rate=48000, category='ui')
        assert [r['output_file'] for r in rows] == ['ui_click_01.wav', 'ui_wind_01.wav'], [r['output_file'] for r in rows]
        y, rate = sf.read(out / 'ui_click_01.wav', always_2d=True)
        assert sf.info(out / 'ui_click_01.wav').subtype == 'PCM_16' and rate == 48000 and y.shape[1] == 2
        assert abs(audio_describe.true_peak(y) - -1.0) <= 0.1, audio_describe.true_peak(y)
        lead = audio_describe.measure(y, rate)['lead_silence_s']
        assert lead <= PRE_ROLL_MS / 1000 + 0.001, f'lead silence {lead} beyond the pre-roll'
        assert rows[0]['licence_files'] == 'Pack/License.txt' and 'trim' in rows[0]['steps'] and rows[0]['source_lead_silence_s'] > 0.19
        # Loudness mode: the 5 s file reaches -20 LUFS; the 0.3 s click is still peak-normalised.
        rows = export(root, ['Pack/Wind Loop.wav', 'Pack/UI/UI_Click_01.wav'], out, fmt='wav24', rate=44100, mode='lufs', lufs=-20)
        y, rate = sf.read(out / rows[0]['output_file'], always_2d=True)
        assert sf.info(out / rows[0]['output_file']).subtype == 'PCM_24'
        assert abs(audio_describe.integrated(audio_describe.loudness_curve(y, rate), rate) - -20) <= 0.2
        assert 'peak' in rows[1]['steps'] and 'loudness' not in rows[1]['steps']
        assert rows[0]['output_file'] == 'sfx_wind_01.wav' and rows[1]['output_file'] == 'sfx_ui_click_01.wav', rows
        # True peak, not sample peak: a quarter-rate tone sampled at 45 degrees peaks 3 dB between its samples.
        n = np.arange(sr)
        sf.write(root / 'Pack/Overs.wav', (0.5 * np.sin(np.pi / 2 * n + np.pi / 4)).astype('float32'), sr)
        rows = export(root, ['Pack/Overs.wav'], out, fmt='wav24', rate=sr, category='test')
        y, _ = sf.read(out / rows[0]['output_file'], always_2d=True)
        assert abs(audio_describe.true_peak(y) - -1.0) <= 0.1 and np.abs(y).max() < 10 ** (-3.5 / 20), audio_describe.true_peak(y)
        rows = export(root, ['Pack/UI/UI_Click_01.wav'], out, fmt='ogg', category='ui')
        assert sf.info(out / rows[0]['output_file']).format == 'OGG' and rows[0]['output_file'] == 'ui_click_01.ogg'
        with (out / 'MANIFEST.csv').open(encoding='utf-8') as f:
            assert len(list(csv.DictReader(f))) == 6, 'manifest keeps every export'
        after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
        assert after == before, 'originals unchanged'
        try:
            export(root, ['Pack/Wind Loop.wav'], root / 'exports')
            raise AssertionError('an output folder inside the library must be refused')
        except ValueError:
            pass
    print('PASS export_sfx check: -1 dBTP peak, lead silence within the 5 ms pre-roll, 16/24-bit WAV and OGG, 48 kHz, '
          '-20 LUFS loudness mode, clean names, manifest with licence, originals unchanged')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cmd', nargs='?', choices=['check'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--list', type=Path, help='one file per line, relative to --root or absolute')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--format', default='wav16', choices=list(FORMATS))
    parser.add_argument('--rate', type=int, default=48000)
    parser.add_argument('--peak', type=float, default=-1.0, help='true-peak target in dBTP (default -1)')
    parser.add_argument('--lufs', type=float, help='integrated loudness target for files of at least --lufs-min-seconds')
    parser.add_argument('--lufs-min-seconds', type=float, default=3.0)
    parser.add_argument('--pattern', default='{category}_{name}_{nn}')
    parser.add_argument('--category', help='category for every file (default: its best agreed studio tag, else sfx)')
    a = parser.parse_args()
    if a.cmd == 'check':
        return check()
    if not (a.root and a.list and a.out):
        parser.error('--root, --list and --out are required')
    rows = export(a.root, read_list(a.list), a.out, a.format, a.rate, 'lufs' if a.lufs is not None else 'peak', a.peak,
                  a.lufs if a.lufs is not None else -18.0, a.lufs_min_seconds, a.pattern, a.category)
    print(json.dumps([{k: r[k] for k in ('output_file', 'source_path', 'steps')} for r in rows], indent=1))


if __name__ == '__main__':
    main()
