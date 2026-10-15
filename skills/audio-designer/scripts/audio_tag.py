"""Recognise what audio contains, and rank files against a description.

    python audio_tag.py tag FILE [FILE ...]            # AudioSet labels per 10 s window, JSON
    python audio_tag.py tag --root D:/lib --only "*bird*" --out tags.csv   # library pass, resumable
    python audio_tag.py rank "soft playful UI confirmation" FILE [FILE ...]
    python audio_tag.py relabel --root D:/lib     # relabel an existing tags CSV with the current tagger
    python audio_tag.py reembed --root D:/lib     # silence-pad short files' fingerprints in place
    python audio_tag.py check

Models download from Hugging Face on first use (cached in ~/.cache/huggingface):
CED-base (527 AudioSet sound-event labels; AST with AUDIO_TAGGER=ast) and LAION CLAP (text-to-audio similarity).
Every tags row records the tagger that wrote it.
A label is a model's reading, not a listen: keep it beside the measurements and the user's ear.
"""

import argparse
import csv
import fnmatch
import json
import os
import sys
from pathlib import Path

import numpy as np

from audio_map import load_audio
from library_metadata import refresh_ledger, write_csv_atomic, metadata_directory, measure_metadata

AST = 'MIT/ast-finetuned-audioset-10-10-0.4593'
# CED runs model code from its Hugging Face repo: pinned to the revision whose code was reviewed (2026-10-11).
TAGGERS = {'ced-base': ('mispeech/ced-base', 'db3e14a8db4c21b56b165261c39649741a900e7f'), 'ast': (AST, None)}
TAGGER = os.environ.get('AUDIO_TAGGER', 'ced-base')
CLAP = 'laion/clap-htsat-unfused'
CLAP_PADDING = 'pad'
CLAP_WINDOW = 10  # seconds: each file's fingerprint covers its first 10 s
AUDIO = {'.wav', '.mp3', '.ogg', '.flac', '.aif', '.aiff', '.m4a', '.aac', '.opus', '.mp4', '.mov'}
FIELDS = ['relative_path', 'size_bytes', 'mtime_ns', 'seconds_tagged', 'speech', 'music',
          'silence', 'top_labels', 'timeline', 'clap_row', 'tagger', 'recognition_status', 'error']
_models = {}


def device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def tagger():
    """The AudioSet tagger named by TAGGER: (feature extractor, model). Both share AudioSet's 527 label names."""
    if 'tagger' not in _models:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        name, revision = TAGGERS[TAGGER]
        remote = TAGGER.startswith('ced')
        _models['tagger'] = (
            AutoFeatureExtractor.from_pretrained(name, revision=revision, trust_remote_code=remote),
            AutoModelForAudioClassification.from_pretrained(name, revision=revision, trust_remote_code=remote).to(device()).eval())
    return _models['tagger']


def clap():
    if 'clap' not in _models:
        from transformers import ClapModel, ClapProcessor
        _models['clap'] = (ClapProcessor.from_pretrained(CLAP),
                           ClapModel.from_pretrained(CLAP).to(device()).eval())
    return _models['clap']


def windows(y, sr, length=10.0):
    size = int(sr * length)
    return [y[i:i + size] for i in range(0, max(len(y), 1), size) if len(y[i:i + size]) >= sr * 0.05] or [y]


def tag_samples(y, sr=16000, window=10.0):
    """Per-window sigmoid probabilities over AudioSet labels (multi-label)."""
    import torch
    extractor, model = tagger()
    # Under a second, pad with silence: CED's first layer needs about 0.2 s, and both models were trained on 10 s clips.
    chunks = [c if len(c) >= sr else np.pad(c, (0, sr - len(c))) for c in windows(y, sr, window)]
    inputs = extractor(chunks, sampling_rate=sr, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model(**inputs).logits
        probs = (out if TAGGER.startswith('ced') else torch.sigmoid(out)).float().cpu().numpy()  # CED outputs probabilities
    return probs, model.config.id2label, window


def summarise(probs, labels, window, top=8, floor=0.1):
    best = probs.max(axis=0)
    order = [i for i in np.argsort(-best) if best[i] >= floor][:top]
    timeline = []
    for n, row in enumerate(probs):
        picks = [i for i in np.argsort(-row)[:3] if row[i] >= floor]
        timeline.append({'start': n * window, 'end': (n + 1) * window,
                         'labels': {labels[i]: round(float(row[i]), 2) for i in picks}})
    named = {v: k for k, v in labels.items()}
    return {'labels': {labels[i]: round(float(best[i]), 2) for i in order},
            **{k.lower(): round(float(best[named[k]]), 2) for k in ('Speech', 'Music', 'Silence')},
            'timeline': timeline}


def recognition_status(y,sr):
    if len(y)/sr<.05:return 'too short to recognise'
    if not np.isfinite(y).all():return 'unreadable'
    if np.max(np.abs(y),initial=0)<=1e-8:return 'digital silence'
    return 'recognised'


def empty_result(status,error=''):
    return dict(labels={},speech=0.,music=0.,silence=1. if status in ('digital silence','silence in analysed excerpt') else 0.,timeline=[],recognition_status=status,error=error)


def tag_file(path, seconds=60):
    try:
        y, sr = load_audio(path, seconds=seconds, sample_rate=16000, mono=True)
    except Exception as exc:
        return {**empty_result('unreadable',str(exc)), 'seconds_tagged':0}
    if not len(y):
        return {**empty_result('unreadable','no decoded samples'), 'seconds_tagged':0}
    status=recognition_status(y,sr)
    if status=='digital silence':
        try:
            if measure_metadata(path)['duration_seconds']>len(y)/sr+.01:status='silence in analysed excerpt'
        except Exception:status='silence in analysed excerpt'
    result = summarise(*tag_samples(y, sr)) if status=='recognised' else empty_result(status)
    result.setdefault('recognition_status',status)
    result['seconds_tagged'] = round(len(y) / sr, 2)
    return result


def embed_audio(ys, sr=48000):
    import torch
    processor, model = clap()
    # Silence-pad short clips: the checkpoint's default 'repeatpad' loops a 0.4 s click into a 10 s rattle.
    inputs = processor(audio=list(ys), sampling_rate=sr, return_tensors='pt', padding=CLAP_PADDING).to(model.device)
    with torch.no_grad():
        e = model.get_audio_features(**inputs)
        e = getattr(e, 'pooler_output', e)  # transformers 5 returns an output object
    return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def embed_text(texts):
    import torch
    processor, model = clap()
    inputs = processor(text=list(texts), return_tensors='pt', padding=True).to(model.device)
    with torch.no_grad():
        e = model.get_text_features(**inputs)
        e = getattr(e, 'pooler_output', e)
    return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def rank(query, paths, seconds=10):
    # ponytail: first `seconds` of each file only, no embedding cache; add one when shortlists pass ~1000 files
    ys = [load_audio(p, seconds=seconds, sample_rate=48000, mono=True)[0] for p in paths]
    scores = embed_audio(ys) @ embed_text([query])[0]
    return sorted(({'path': str(p), 'score': round(float(s), 3)} for p, s in zip(paths, scores)),
                  key=lambda r: -r['score'])


def decode(path, seconds):
    """One decode at CLAP's 48 kHz; AST gets a 3:1 polyphase resample of the same samples."""
    from scipy.signal import resample_poly
    y48, _ = load_audio(path, seconds=seconds, sample_rate=48000, mono=True)
    if not len(y48):
        raise ValueError('empty audio')
    return y48, resample_poly(y48, 1, 3).astype('float32')


def tag_row(rel, size, mtime, r, seconds_tagged, clap_row, tagger_name):
    return {'relative_path': rel, 'size_bytes': size, 'mtime_ns': mtime, 'seconds_tagged': seconds_tagged,
            'speech': r['speech'], 'music': r['music'], 'silence': r['silence'],
            'top_labels': '; '.join(f'{k} {v}' for k, v in r['labels'].items()),
            'timeline': ' | '.join(f"{w['start']:g}-{w['end']:g}: " + ', '.join(f'{k} {v}' for k, v in w['labels'].items())
                                   for w in r['timeline']),
            'clap_row': clap_row, 'tagger': tagger_name, 'recognition_status':r.get('recognition_status','recognised'), 'error':r.get('error','')}


def library(root, only, out, seconds, batch=16):
    """Resumable pass: tags to `out` (CSV), CLAP vectors to `<out>_clap.f32` (512 float32 per row,
    row index in `clap_row`). A changed file appends a new row; readers keep the last row per path."""
    from concurrent.futures import ThreadPoolExecutor
    root, out = Path(root), Path(out)
    if not root.is_dir():
        raise ValueError(f'Library folder does not exist: {root}')
    if out.name=='AUDIO_TAGS.csv' and out.parent.resolve()==root.resolve():
        out=metadata_directory(root,create=True)/out.name
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        with out.open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            if 'tagger' not in (reader.fieldnames or []):
                sys.exit(f'{out} predates the tagger column: run `audio_tag.py relabel --root {root}` first')
            done = {r['relative_path']: r for r in reader}
    all_files = sorted(p for p in root.rglob('*') if p.suffix.lower() in AUDIO and p.is_file())
    # Full-length metadata is independent of the recognition excerpt and --only selection.
    refresh_ledger(root, all_files, metadata_root=out.parent)
    with (out.parent/'AUDIO_LEDGER_files.csv').open(encoding='utf-8-sig',newline='') as ledger:
        durations={r['relative_path']:float(r.get('duration_seconds') or 0) for r in csv.DictReader(ledger)}
    files = [p for p in all_files
             if fnmatch.fnmatch(p.relative_to(root).as_posix().lower(), (only or '*').lower())]
    todo = []
    for path in files:
        rel, stat = path.relative_to(root).as_posix(), path.stat()
        old = done.get(rel)
        if not (old and old.get('recognition_status')!='unreadable' and old['size_bytes'] == str(stat.st_size) and old['mtime_ns'] == str(stat.st_mtime_ns)):
            todo.append((path, rel, stat))
    print(f'{len(files)} files, {len(todo)} to tag', file=sys.stderr, flush=True)
    vectors = out.with_name(out.stem + '_clap.f32')
    new = not out.exists()
    tagged, skipped = 0, []
    with out.open('a', encoding='utf-8', newline='') as f, vectors.open('ab') as v, ThreadPoolExecutor(4) as pool:
        writer = csv.DictWriter(f, FIELDS)
        if new:
            writer.writeheader()
        for b in range(0, len(todo), batch):
            chunk = todo[b:b + batch]
            decoded = list(pool.map(lambda item: _try(decode, item[0], seconds), chunk))
            for (path, rel, stat), audio in zip(chunk, decoded):
                try:
                    if isinstance(audio, Exception):
                        raise audio
                    y48, y16 = audio
                    status=recognition_status(y16,16000)
                    if status=='digital silence' and durations.get(rel,0)>len(y16)/16000+.01:status='silence in analysed excerpt'
                    r = summarise(*tag_samples(y16, 16000)) if status=='recognised' else empty_result(status)
                    r.setdefault('recognition_status',status)
                    vec = embed_audio([y48[:48000 * 10]])[0].astype('<f4') if status=='recognised' else None
                except Exception as exc:  # one bad file must not stop a 35k-file pass
                    reason = str(exc).strip() or type(exc).__name__
                    skipped.append({'relative_path': rel, 'reason': reason})
                    print(f'SKIP {rel}: {reason}', file=sys.stderr, flush=True)
                    previous=done.get(rel,{})
                    if previous.get('recognition_status')!='unreadable' or previous.get('size_bytes')!=str(stat.st_size) or previous.get('mtime_ns')!=str(stat.st_mtime_ns):
                        writer.writerow(tag_row(rel,stat.st_size,stat.st_mtime_ns,empty_result('unreadable',reason),0,-1,TAGGER));f.flush()
                    continue
                row = v.tell() // (4 * len(vec)) if vec is not None else -1
                if vec is not None:v.write(vec.tobytes())
                v.flush()
                settings = clap_settings(vectors)
                if settings['pad_checked_rows'] == row:  # every earlier row is known to be silence-padded
                    settings['pad_checked_rows'] = row + 1
                    save_clap_settings(vectors, settings)
                writer.writerow(tag_row(rel, stat.st_size, stat.st_mtime_ns, r, round(len(y16) / 16000, 2), row, TAGGER))
                f.flush()
                tagged += 1
            print(f'{min(b + batch, len(todo))}/{len(todo)}', file=sys.stderr, flush=True)
    write_csv_atomic(out.parent / 'AUDIO_TAGS_skipped.csv', ['relative_path', 'reason'], skipped)
    summary = {'tagged': tagged, 'unchanged': len(files) - len(todo), 'skipped': len(skipped)}
    print('SUMMARY ' + ' '.join(f'{key}={value}' for key, value in summary.items()), file=sys.stderr, flush=True)
    return summary


def clap_settings(vectors):
    """How the fingerprints in `vectors` were made. Rows before pad_checked_rows are known to be silence-padded."""
    path = vectors.with_suffix('.json')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {'model': CLAP, 'window_s': CLAP_WINDOW, 'short_clip_padding': CLAP_PADDING, 'pad_checked_rows': 0}


def save_clap_settings(vectors, settings):
    tmp = vectors.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(settings, indent=1), encoding='utf-8')
    tmp.replace(vectors.with_suffix('.json'))


def reembed(root, tags_csv, batch=32):
    """Re-fingerprint short files (under CLAP_WINDOW seconds) in place with silence padding.
    Longer files are never padded, so their rows are only marked checked. Resumable, and safe to rerun
    after a pass that still used looping: it continues from the settings file's pad_checked_rows."""
    from concurrent.futures import ThreadPoolExecutor
    vectors = tags_csv.with_name(tags_csv.stem + '_clap.f32')
    settings = clap_settings(vectors)
    total = vectors.stat().st_size // (4 * 512)
    start = settings['pad_checked_rows']
    owner = {}
    with tags_csv.open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if r.get('clap_row'):
                owner[int(r['clap_row'])] = (r['relative_path'], float(r['seconds_tagged'] or 0))
    short = [(row, owner[row][0]) for row in range(start, total) if row in owner and owner[row][1] < CLAP_WINDOW]
    print(f'rows {start}-{total}: {len(short)} short files to re-fingerprint, the rest are checked as-is', file=sys.stderr, flush=True)
    with vectors.open('r+b') as v, ThreadPoolExecutor(3) as pool:
        for b in range(0, len(short), batch):
            chunk = short[b:b + batch]
            decoded = list(pool.map(lambda item: _try(lambda p: load_audio(root / p, seconds=CLAP_WINDOW, sample_rate=48000,
                                                                          mono=True)[0], item[1]), chunk))
            good = [(row, y) for (row, _), y in zip(chunk, decoded) if not isinstance(y, Exception) and len(y)]
            if good:
                for (row, _), vec in zip(good, embed_audio([y for _, y in good])):
                    v.seek(row * 4 * 512)
                    v.write(vec.astype('<f4').tobytes())
                v.flush()
            print(f'{min(b + batch, len(short))}/{len(short)}', file=sys.stderr, flush=True)
    settings.update({'model': CLAP, 'window_s': CLAP_WINDOW, 'short_clip_padding': CLAP_PADDING, 'pad_checked_rows': total})
    save_clap_settings(vectors, settings)
    print(f'pad_checked_rows = {total}', file=sys.stderr)


def relabel(root, tags_csv, seconds=60, batch=16):
    """Relabel every file in an existing tags CSV with the current tagger. Fingerprints, clap_row, size and mtime
    are kept. Writes <csv>.relabel.csv as it goes (resumable), then swaps it in and keeps the old file as
    <csv>.before-relabel.csv. A file that cannot be decoded keeps its old labels and its old tagger name."""
    from concurrent.futures import ThreadPoolExecutor
    with tags_csv.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        old_tagger = 'ast' if 'tagger' not in (reader.fieldnames or []) else None
        rows = {r['relative_path']: r for r in reader}
    side = tags_csv.with_suffix('.relabel.csv')
    done = set()
    if side.exists():
        with side.open(encoding='utf-8', newline='') as f:
            done = {r['relative_path'] for r in csv.DictReader(f)}
    todo = [r for rel, r in rows.items() if rel not in done]
    print(f'{len(rows)} rows, {len(todo)} to relabel with {TAGGER}', file=sys.stderr, flush=True)
    new = not side.exists()
    with side.open('a', encoding='utf-8', newline='') as f, ThreadPoolExecutor(4) as pool:
        writer = csv.DictWriter(f, FIELDS)
        if new:
            writer.writeheader()
        for b in range(0, len(todo), batch):
            chunk = todo[b:b + batch]
            decoded = list(pool.map(lambda r: _try(lambda p: load_audio(root / p, seconds=seconds, sample_rate=16000,
                                                                         mono=True)[0], r['relative_path']), chunk))
            for old, y in zip(chunk, decoded):
                if isinstance(y, Exception) or not len(y):
                    print(f'KEEP OLD {old["relative_path"]}: {y if isinstance(y, Exception) else "empty"}', file=sys.stderr, flush=True)
                    writer.writerow({k: old.get(k, '') for k in FIELDS} | {'tagger': old.get('tagger') or old_tagger})
                    continue
                r = summarise(*tag_samples(y, 16000))
                writer.writerow(tag_row(old['relative_path'], old['size_bytes'], old['mtime_ns'], r,
                                        round(len(y) / 16000, 2), old['clap_row'], TAGGER))
            f.flush()
            print(f'{min(b + batch, len(todo))}/{len(todo)}', file=sys.stderr, flush=True)
    backup = tags_csv.with_suffix('.before-relabel.csv')
    tags_csv.replace(backup)
    side.replace(tags_csv)
    print(f'{tags_csv} relabelled with {TAGGER}; previous labels kept in {backup}', file=sys.stderr)


def _try(fn, *args):
    try:
        return fn(*args)
    except Exception as exc:
        return exc


def check():
    sr = 16000
    t = np.arange(sr * 4) / sr
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype('float32')
    noise = (0.3 * np.random.default_rng(1).standard_normal(sr * 4)).astype('float32')
    tone_labels = summarise(*tag_samples(tone, sr))['labels']
    noise_labels = summarise(*tag_samples(noise, sr))['labels']
    assert any(k in tone_labels for k in ('Sine wave', 'Tuning fork', 'Beep, bleep')), tone_labels
    assert any('noise' in k.lower() or k in ('Static', 'Hiss') for k in noise_labels), noise_labels
    up = lambda y: np.interp(np.arange(len(y) * 3) / 3, np.arange(len(y)), y).astype('float32')
    audio = embed_audio([up(tone), up(noise)])
    text = embed_text(['a steady pure sine tone', 'loud white noise hiss'])
    sims = audio @ text.T
    assert sims[0, 0] > sims[1, 0] and sims[1, 1] > sims[0, 1], sims
    reembed_check()
    relabel_check()
    print(f'PASS audio_tag check on {device()}: tone {list(tone_labels)[:2]}, noise {list(noise_labels)[:2]}, CLAP ranks both, '
          'reembed rewrites only short rows')


def relabel_check():
    """An old-format CSV (no tagger column, junk labels) must come back with current labels, same clap_row."""
    import tempfile
    import soundfile as sf
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sr = 16000
        sf.write(root / 'tone.wav', (0.3 * np.sin(2 * np.pi * 440 * np.arange(sr * 3) / sr)).astype('float32'), sr)
        out = root / 'AUDIO_TAGS.csv'
        old_fields = [f for f in FIELDS if f != 'tagger']
        with out.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, old_fields)
            w.writeheader()
            w.writerow({'relative_path': 'tone.wav', 'size_bytes': 1, 'mtime_ns': 2, 'seconds_tagged': 3, 'speech': 0,
                        'music': 0, 'silence': 0, 'top_labels': 'Junk 0.9', 'timeline': '0-10: Junk 0.9', 'clap_row': 7})
            w.writerow({'relative_path': 'missing.wav', 'size_bytes': 1, 'mtime_ns': 2, 'seconds_tagged': 3, 'speech': 0,
                        'music': 0, 'silence': 0, 'top_labels': 'Kept 0.5', 'timeline': '', 'clap_row': 8})
        try:
            library(root, None, out, 10)
            raise AssertionError('a pass must refuse a CSV without the tagger column')
        except SystemExit:
            pass
        relabel(root, out)
        rows = {r['relative_path']: r for r in csv.DictReader(out.open(encoding='utf-8'))}
        assert rows['tone.wav']['tagger'] == TAGGER and rows['tone.wav']['clap_row'] == '7', rows['tone.wav']
        assert 'Sine wave' in rows['tone.wav']['top_labels'], rows['tone.wav']['top_labels']
        assert rows['missing.wav']['top_labels'] == 'Kept 0.5' and rows['missing.wav']['tagger'] == 'ast', rows['missing.wav']
        assert (root / 'AUDIO_TAGS.before-relabel.csv').exists() and not (root / 'AUDIO_TAGS.relabel.csv').exists()


def reembed_check():
    """A library pass, a short row overwritten with a looped fingerprint, then reembed must restore silence padding."""
    import tempfile
    import soundfile as sf
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sr = 48000
        click = np.zeros(sr // 2, 'float32')
        click[100:400] = np.hanning(300) * 0.8
        sf.write(root / 'click.wav', click, sr)
        sf.write(root / 'tone.wav', (0.2 * np.sin(2 * np.pi * 330 * np.arange(sr * 12) / sr)).astype('float32'), sr)
        out = root / 'AUDIO_TAGS.csv'
        library(root, None, out, 12)
        vectors = out.with_name('AUDIO_TAGS_clap.f32')
        assert clap_settings(vectors)['pad_checked_rows'] == 2, clap_settings(vectors)
        rows = {r['relative_path']: int(r['clap_row']) for r in csv.DictReader(out.open(encoding='utf-8'))}
        m = np.fromfile(vectors, '<f4').reshape(-1, 512).copy()
        padded, long_row = m[rows['click.wav']].copy(), m[rows['tone.wav']].copy()
        # Simulate a row written by the old looping code, and a settings file that has not checked it.
        import torch
        processor, model = clap()
        with torch.no_grad():
            x = processor(audio=[click], sampling_rate=sr, return_tensors='pt', padding='repeatpad').to(model.device)
            e = model.get_audio_features(**x)
            looped = torch.nn.functional.normalize(getattr(e, 'pooler_output', e), dim=-1).cpu().numpy()[0]
        assert float(looped @ padded) < 0.999, 'looping and padding should give different fingerprints'
        m[rows['click.wav']] = looped
        m.astype('<f4').tofile(vectors)
        save_clap_settings(vectors, dict(clap_settings(vectors), pad_checked_rows=0))
        reembed(root, out)
        after = np.fromfile(vectors, '<f4').reshape(-1, 512)
        assert float(after[rows['click.wav']] @ padded) > 0.999, 'short row not restored to silence padding'
        assert np.array_equal(after[rows['tone.wav']], long_row), 'long row must be left as-is'
        assert clap_settings(vectors)['pad_checked_rows'] == 2


def main():
    parser = argparse.ArgumentParser(
        description='Recognise audio events and rank files against a description.',
        epilog='Example: python scripts/audio_tag.py tag --root MySounds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest='cmd', required=True, title='commands', help='available commands')
    t = sub.add_parser(
        'tag',
        description='Label audio files and cache fingerprints; library passes resume unchanged files.',
        epilog='Example: python scripts/audio_tag.py tag --root MySounds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help='Label audio files and cache fingerprints; library passes resume unchanged files.',
    )
    t.add_argument('files', nargs='*', type=Path, help='audio files to analyse')
    t.add_argument('--root', type=Path, help='folder containing the audio library')
    t.add_argument('--only', help='relative-path glob under --root')
    t.add_argument('--out', type=Path, help='CSV for a resumable library pass (needs --root)')
    t.add_argument('--seconds', type=float, default=60, help='maximum opening audio span to analyse, in seconds')
    r = sub.add_parser(
        'rank',
        description='Rank supplied files by audio similarity to a text description.',
        epilog='Example: python scripts/audio_tag.py rank "soft wooden click" click.wav knock.wav',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help='Rank supplied files by audio similarity to a text description.',
    )
    r.add_argument('query', help='text description of the wanted sound')
    r.add_argument('files', nargs='+', type=Path, help='audio files to analyse')
    r.add_argument('--seconds', type=float, default=10, help='maximum opening audio span to analyse, in seconds')
    rl = sub.add_parser(
        'relabel',
        help='relabel an existing tags CSV with the current tagger, keeping fingerprints',
        description='Replace labels in an existing tags CSV, preserving its fingerprints.',
        epilog='Example: python scripts/audio_tag.py relabel --root MySounds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rl.add_argument('--root', type=Path, required=True, help='folder containing the audio library')
    rl.add_argument('--tags', type=Path, help='tags CSV (default <root>/AUDIO_TAGS.csv)')
    e = sub.add_parser(
        'reembed',
        help='re-fingerprint short files with silence padding, in place',
        description='Replace short-file fingerprints using silence padding.',
        epilog='Example: python scripts/audio_tag.py reembed --root MySounds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    e.add_argument('--root', type=Path, required=True, help='folder containing the audio library')
    e.add_argument('--tags', type=Path, help='tags CSV (default <root>/AUDIO_TAGS.csv)')
    sub.add_parser(
        'check',
        description='Run synthetic checks without a library or model download.',
        epilog='Example: python scripts/audio_tag.py check',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help='Run synthetic checks without a library or model download.',
    )
    a = parser.parse_args()
    if a.cmd == 'relabel':
        return relabel(a.root, a.tags or a.root / 'AUDIO_TAGS.csv')
    if a.cmd == 'reembed':
        return reembed(a.root, a.tags or a.root / 'AUDIO_TAGS.csv')
    if a.cmd == 'check':
        return check()
    if a.cmd == 'rank':
        return print(json.dumps(rank(a.query, a.files, a.seconds), indent=1))
    if a.out and not a.root:
        parser.error('--out needs --root')
    if a.root:
        if a.files:
            parser.error('Use file arguments or --root, not both')
        return library(a.root, a.only, a.out or a.root / 'AUDIO_TAGS.csv', a.seconds)
    if not a.files:
        parser.error('give audio files or --root')
    files = a.files
    print(json.dumps({str(p): tag_file(p, a.seconds) for p in files}, indent=1))


if __name__ == '__main__':
    main()
