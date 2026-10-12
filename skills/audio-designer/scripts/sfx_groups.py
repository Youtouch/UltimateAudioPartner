"""Duplicates and variation sets: one row per sound you actually have, and the different takes of one sound.

    python sfx_groups.py build --root D:/MySounds              # -> AUDIO_GROUPS.csv in the root
    python sfx_groups.py varied --root D:/MySounds --path "Pack/UI/Click_01.wav" --n 4
    python sfx_groups.py check

Duplicates: the same sound twice (a WAV and its MP3, one file in two packs). Candidates are files whose audio
fingerprints are close (cosine >= DUP_COSINE) and whose sound lengths agree (DUP_SECONDS, allowing an MP3's
encoder padding); a candidate pair is a duplicate only when the waveforms themselves match (normalised
cross-correlation >= DUP_WAVEFORM within 10 ms of lag, after aligning their first sounds), because separate takes of one sound can share a
fingerprint. Level does not matter: a louder copy is still a copy. The kept copy prefers a lossless file,
then the shortest path.

Variation sets: takes of one sound for round-robin playback. Files in one folder whose names differ only by a
trailing number or letter (Click_01 .. Click_30), or, failing a name, a tight fingerprint cluster in one folder.
`varied` returns the N most different members, by fingerprint and measured descriptors, for round-robin use.
Needs AUDIO_TAGS.csv and its fingerprints; AUDIO_DESCRIPTORS.csv (audio_describe.py) sharpens both.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from library_metadata import read_metadata, relative_path, write_csv_atomic
import audio_describe

DIM = 512
DUP_COSINE = 0.98          # fingerprint candidates; the self-test's 192 kbps MP3 copy passes it
DUP_SECONDS = (0.06, 0.02)  # sound lengths agree within 60 ms or 2%, whichever is larger
DUP_WAVEFORM = 0.97         # normalised waveform cross-correlation, first 2 s at 16 kHz (a 192k MP3 copy: 0.997)
CLUSTER_COSINE = 0.93       # fingerprint cluster in one folder, for files whose names carry no take number
LOSSLESS = {'.wav', '.flac', '.aif', '.aiff'}
TAKE = re.compile(r'(?:[\s_\-.]*(?:v|var|take|alt|version)?[\s_\-.]*\d{1,3}[a-z]?|[\s_\-.]+[a-z])$', re.I)
FIELDS = ['relative_path', 'duplicate_of', 'duplicate_set', 'variation_set', 'variation_size']


def stem_of(path):
    """'Pack/UI/Click_07.wav' -> ('Pack/UI', 'click'); a name that is only a number keeps it."""
    folder, _, name = relative_path(path).rpartition('/')
    base = name.rsplit('.', 1)[0]
    stem = base
    for _ in range(2):  # 'Hit_03_B' -> 'Hit'
        cut = TAKE.sub('', stem)
        stem = cut if cut.strip(' _-.') else stem
    return folder, stem.strip(' _-.').lower()


def load(root, descriptors=None):
    """Paths with a fingerprint, their unit vectors, sound lengths and the descriptor table (None when unmeasured)."""
    root = Path(root)
    tags = {}
    with (root / 'AUDIO_TAGS.csv').open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            tags[relative_path(r['relative_path'])] = r
    vectors = np.fromfile(root / 'AUDIO_TAGS_clap.f32', '<f4')
    vectors = vectors[:len(vectors) // DIM * DIM].reshape(-1, DIM)
    paths, rows = [], []
    for p, r in tags.items():
        row = int(r['clap_row']) if r.get('clap_row') else -1
        if 0 <= row < len(vectors) and (root / p).exists():
            paths.append(p)
            rows.append(row)
    audio = vectors[rows] if rows else np.zeros((0, DIM), '<f4')
    meta = read_metadata(root, tags)
    desc = audio_describe.read_descriptors(Path(descriptors).parent if descriptors else root)
    seconds = np.array([_f(desc.get(p, {}).get('sound_s')) for p in paths])
    full = np.array([_f(meta.get(p, {}).get('duration_seconds')) for p in paths])
    seconds = np.where(np.isfinite(seconds), seconds, full)
    table = audio_describe.table_for(paths, desc) if desc else None
    return paths, audio, seconds, table


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def components(n, pairs):
    parent = list(range(n))
    for a, b in pairs:
        ra, rb = _find(parent, a), _find(parent, b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = defaultdict(list)
    for i in range(n):
        groups[_find(parent, i)].append(i)
    return [g for g in groups.values() if len(g) > 1]


def similar_pairs(audio, threshold, allowed=None, chunk=2048):
    """Index pairs (i < j) with cosine >= threshold; `allowed(i_block, j)` filters further. Chunked: memory stays flat."""
    out = []
    for start in range(0, len(audio), chunk):
        sims = audio[start:start + chunk] @ audio.T
        i, j = np.nonzero(sims >= threshold)
        i = i + start
        keep = i < j
        for a, b in zip(i[keep], j[keep]):
            if allowed is None or allowed(a, b):
                out.append((int(a), int(b)))
    return out


def waveform_match(x, y, max_lag=160):
    """Peak normalised cross-correlation of two signals within max_lag samples either way (FFT)."""
    size = 1 << (len(x) + len(y) - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(x, size) * np.conj(np.fft.rfft(y, size)), size)
    c = np.concatenate([c[-max_lag:], c[:max_lag + 1]])
    return float(np.abs(c).max() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def duplicates(root, paths, audio, seconds):
    from functools import lru_cache
    from audio_map import load_audio

    @lru_cache(maxsize=4096)
    def wave(i):
        try:
            y = load_audio(Path(root) / paths[i], seconds=2.0, sample_rate=16000, mono=True)[0].astype(np.float32)
        except Exception:
            return None
        start = np.flatnonzero(np.abs(y) > 1e-3)
        return y[start[0]:] if len(start) else None  # align starts: an MP3's encoder padding shifts it

    def same(a, b):
        x, y = wave(a), wave(b)
        return x is not None and y is not None and waveform_match(x, y) >= DUP_WAVEFORM

    # Candidates: close fingerprints and agreeing lengths, checked in bulk; the closest pairs are decoded first, and a
    # pair already joined through a third copy is never decoded (decoding scattered files is the slow part).
    candidates = []
    for start in range(0, len(audio), 2048):
        sims = audio[start:start + 2048] @ audio.T
        i, j = np.nonzero(sims >= DUP_COSINE)
        cos, i = sims[i, j], i + start
        sa, sb = seconds[i], seconds[j]
        with np.errstate(invalid='ignore'):
            ok = (i < j) & (np.abs(sa - sb) <= np.maximum(DUP_SECONDS[0], DUP_SECONDS[1] * np.maximum(sa, sb)))
        candidates += zip(-cos[ok], i[ok].tolist(), j[ok].tolist())
    candidates.sort()
    parent, checked = list(range(len(paths))), 0
    for _, a, b in candidates:
        ra, rb = _find(parent, a), _find(parent, b)
        if ra != rb:
            checked += 1
            if same(a, b):
                parent[max(ra, rb)] = min(ra, rb)
    print(f'{len(candidates)} candidate pairs, {checked} decoded and compared', file=sys.stderr, flush=True)
    groups = defaultdict(list)
    for i in range(len(paths)):
        groups[_find(parent, i)].append(i)
    sets = [g for g in groups.values() if len(g) > 1]
    key = lambda i: (Path(paths[i]).suffix.lower() not in LOSSLESS, len(paths[i]), paths[i])
    return [sorted(s, key=key) for s in sets]  # first member is the copy to keep


def variations(paths, audio, seconds, hidden):
    """Name stems first; then fingerprint clusters among the remaining files of each folder. Duplicates (hidden) are left out."""
    by_stem = defaultdict(list)
    for i, p in enumerate(paths):
        if i not in hidden:
            by_stem[stem_of(p)].append(i)
    sets = [sorted(m, key=lambda i: paths[i]) for m in by_stem.values() if len(m) > 1]
    named = {i for s in sets for i in s}
    by_folder = defaultdict(list)
    for i, p in enumerate(paths):
        if i not in hidden and i not in named:
            by_folder[relative_path(p).rpartition('/')[0]].append(i)
    for members in by_folder.values():
        if len(members) < 3:
            continue
        m = np.array(members)
        sims = audio[m] @ audio[m].T
        ratio = lambda a, b: max(seconds[a], seconds[b]) / max(1e-3, min(seconds[a], seconds[b]))
        pairs = [(a, b) for a, b in zip(*np.nonzero(np.triu(sims >= CLUSTER_COSINE, 1))) if ratio(m[a], m[b]) <= 1.5]
        sets += [sorted((int(m[k]) for k in c), key=lambda i: paths[i]) for c in components(len(m), pairs) if len(c) >= 3]
    return sets


def build(root, out=None, descriptors=None):
    paths, audio, seconds, table = load(root, descriptors)
    dups = duplicates(root, paths, audio, seconds)
    hidden = {i for s in dups for i in s[1:]}
    vars_ = variations(paths, audio, seconds, hidden)
    rows = {p: dict.fromkeys(FIELDS, '') | {'relative_path': p} for p in paths}
    for n, s in enumerate(dups, 1):
        for i in s:
            rows[paths[i]].update(duplicate_set=f'd{n}', duplicate_of=paths[s[0]] if i != s[0] else '')
    for n, s in enumerate(vars_, 1):
        for i in s:
            rows[paths[i]].update(variation_set=f'v{n}', variation_size=len(s))
    out_path, rows_out = out, [r for r in rows.values() if r['duplicate_set'] or r['variation_set']]
    write_csv_atomic(Path(out_path) if out_path else Path(root) / 'AUDIO_GROUPS.csv', FIELDS, rows_out)
    summary = {'files': len(paths), 'duplicate_sets': len(dups), 'duplicate_copies_hidden': len(hidden),
               'variation_sets': len(vars_), 'files_in_variation_sets': sum(len(s) for s in vars_)}
    print(json.dumps(summary))
    return summary


def read_groups(root):
    path = Path(root) / 'AUDIO_GROUPS.csv'
    if not path.exists():
        return {}
    with path.open(encoding='utf-8', newline='') as f:
        return {relative_path(r['relative_path']): r for r in csv.DictReader(f)}


VARIED_MEASURES = ('sound_s', 'centroid_hz', 'pitch_hz', 'lufs_momentary_max')  # what a listener hears change between takes


def pick_varied(indices, audio, table, n):
    """Greedy farthest-point choice: start from the most typical member, then repeatedly add the member farthest
    from everything chosen. Distance = fingerprint distance (1 - cosine) plus the distance between measured
    length, brightness, pitch and loudness (z-scored, log scale except LUFS), each divided by its median so
    neither dominates."""
    idx = list(indices)
    n = min(n, len(idx))
    off = ~np.eye(len(idx), dtype=bool)
    scaled = lambda d: d / (np.median(d[off]) or 1.0) if len(idx) > 1 else d
    dist = scaled(1 - audio[idx] @ audio[idx].T)
    if table is not None:
        cols = []
        for measure in VARIED_MEASURES:
            x = table[measure][idx].astype(float)
            x = x if measure == 'lufs_momentary_max' else np.log(np.maximum(x, 1e-6))
            if np.isfinite(x).sum() >= 2 and np.nanstd(x) > 0:
                cols.append(np.nan_to_num((x - np.nanmean(x)) / np.nanstd(x)))
        if cols:
            f = np.stack(cols, axis=1)
            dist = dist + scaled(np.sqrt(((f[:, None] - f[None]) ** 2).sum(-1)))
    chosen = [int(np.argmin(dist.sum(axis=1)))]
    while len(chosen) < n:
        gap = dist[:, chosen].min(axis=1)
        gap[chosen] = -1
        chosen.append(int(np.argmax(gap)))
    return [idx[c] for c in chosen]


def varied(root, path, n):
    paths, audio, _, table = load(root)
    groups = read_groups(root)
    wanted = groups.get(relative_path(path), {}).get('variation_set')
    if not wanted:
        raise ValueError(f'{path} is not in a variation set (run sfx_groups.py build first)')
    members = [i for i, p in enumerate(paths) if groups.get(p, {}).get('variation_set') == wanted]
    return [paths[i] for i in pick_varied(members, audio, table, n)]


def check():
    """A synthetic library: a WAV with its MP3 and a copy in a second pack (one duplicate set), and Click_01..10 built
    from three distinct clicks (one variation set); pick 3 varied must return one click of each kind."""
    import shutil
    import subprocess
    import tempfile
    import soundfile as sf
    from scipy.signal import butter, sosfilt
    from audio_map import media_tool
    from audio_tag import library as tag_library
    assert stem_of('Pack/UI/Click_07.wav') == ('Pack/UI', 'click') and stem_of('P/Hit_03_B.wav') == ('P', 'hit')
    assert stem_of('P/Sword Swing 2.wav') == ('P', 'sword swing') and stem_of('P/01.wav') == ('P', '01')
    assert stem_of('P/Laser_Take3.wav') == ('P', 'laser') and stem_of('P/Boom.wav') == ('P', 'boom')
    sr, rng = 48000, np.random.default_rng(5)
    t = np.arange(int(0.25 * sr)) / sr
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for folder in ('PackA/Doors', 'PackB/Doors', 'PackA/UI', 'PackA/Misc'):
            (root / folder).mkdir(parents=True)
        door = np.concatenate([rng.standard_normal(int(0.6 * sr)) * np.exp(-np.arange(int(0.6 * sr)) / (0.15 * sr)) * 0.5,
                               np.zeros(int(0.1 * sr))])
        sf.write(root / 'PackA/Doors/Door_Slam.wav', door.astype('float32'), sr)
        shutil.copy(root / 'PackA/Doors/Door_Slam.wav', root / 'PackB/Doors/Heavy Door Slam.wav')
        subprocess.run([media_tool(), '-v', 'error', '-i', str(root / 'PackA/Doors/Door_Slam.wav'), '-b:a', '192k',
                        str(root / 'PackA/Doors/Door_Slam.mp3')], check=True)
        kinds = {0: 400, 1: 1500, 2: 5000}  # three kinds of click, by pitch; each take differs a little in pitch and decay
        for k in range(10):
            f = kinds[k % 3] * (1 + 0.03 * rng.standard_normal())
            noise = sosfilt(butter(2, [f / 1.5, f * 1.5], 'bandpass', fs=sr, output='sos'), rng.standard_normal(len(t)))
            body = np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) + 2 * noise  # each take its own noise, in the click's band
            y = body * np.exp(-t / rng.uniform(0.02, 0.04)) * 0.4
            sf.write(root / f'PackA/UI/Click_{k + 1:02d}.wav', y.astype('float32'), sr)
        for name, f in (('Whistle.wav', 900), ('Hum.wav', 60), ('Siren.wav', 700)):  # unrelated singles
            y = 0.3 * np.sin(2 * np.pi * f * np.arange(sr) / sr * (1 + (name == 'Siren.wav') * np.arange(sr) / sr))
            sf.write(root / f'PackA/Misc/{name}', y.astype('float32'), sr)
        tag_library(root, None, root / 'AUDIO_TAGS.csv', 10)
        audio_describe.library(root, workers=1)
        summary = build(root)
        groups = read_groups(root)
        doors = {p for p, g in groups.items() if g['duplicate_set']}
        assert doors == {'PackA/Doors/Door_Slam.wav', 'PackA/Doors/Door_Slam.mp3', 'PackB/Doors/Heavy Door Slam.wav'}, groups
        kept = [p for p in doors if not groups[p]['duplicate_of']]
        assert kept == ['PackA/Doors/Door_Slam.wav'], 'the lossless copy with the shortest path is kept'
        clicks = {p for p, g in groups.items() if g['variation_set']}
        assert clicks == {f'PackA/UI/Click_{k:02d}.wav' for k in range(1, 11)}, clicks
        assert summary['duplicate_sets'] == 1 and summary['variation_sets'] == 1, summary
        picked = varied(root, 'PackA/UI/Click_04.wav', 3)
        assert sorted((int(p[-6:-4]) - 1) % 3 for p in picked) == [0, 1, 2], picked
        assert len(varied(root, 'PackA/UI/Click_04.wav', 20)) == 10
    print('PASS sfx_groups check: WAV/MP3/second-pack duplicates in one set, lossless copy kept, Click_01..10 one set, '
          'pick 3 varied returns one of each kind')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--root', type=Path, required=True)
    b.add_argument('--out', type=Path, help='CSV (default <root>/AUDIO_GROUPS.csv)')
    b.add_argument('--descriptors', type=Path, help='AUDIO_DESCRIPTORS.csv kept elsewhere (default: in the root)')
    v = sub.add_parser('varied', help='the N most different members of a file\'s variation set')
    v.add_argument('--root', type=Path, required=True)
    v.add_argument('--path', required=True)
    v.add_argument('--n', type=int, default=4)
    sub.add_parser('check')
    a = parser.parse_args()
    if a.cmd == 'check':
        return check()
    if a.cmd == 'build':
        return build(a.root, a.out, a.descriptors)
    print(json.dumps(varied(a.root, a.path, a.n), indent=1))


if __name__ == '__main__':
    main()
