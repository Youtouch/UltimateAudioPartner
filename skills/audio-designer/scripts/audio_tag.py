"""Recognise what audio contains, and rank files against a description.

    python audio_tag.py tag FILE [FILE ...]            # AudioSet labels per 10 s window, JSON
    python audio_tag.py tag --root D:/lib --only "*bird*" --out tags.csv   # library pass, resumable
    python audio_tag.py rank "soft playful UI confirmation" FILE [FILE ...]
    python audio_tag.py check

Models download from Hugging Face on first use (cached in ~/.cache/huggingface):
AST fine-tuned on AudioSet (527 sound-event labels) and LAION CLAP (text-to-audio similarity).
A label is a model's reading, not a listen: keep it beside the measurements and the user's ear.
"""

import argparse
import csv
import fnmatch
import json
import sys
from pathlib import Path

import numpy as np

from audio_map import load_audio

AST = 'MIT/ast-finetuned-audioset-10-10-0.4593'
CLAP = 'laion/clap-htsat-unfused'
AUDIO = {'.wav', '.mp3', '.ogg', '.flac', '.aif', '.aiff', '.m4a', '.aac', '.opus', '.mp4', '.mov'}
FIELDS = ['relative_path', 'size_bytes', 'mtime_ns', 'seconds_tagged', 'speech', 'music',
          'silence', 'top_labels', 'timeline', 'clap_row']
_models = {}


def device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def ast():
    if 'ast' not in _models:
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        _models['ast'] = (ASTFeatureExtractor.from_pretrained(AST),
                          ASTForAudioClassification.from_pretrained(AST).to(device()).eval())
    return _models['ast']


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
    extractor, model = ast()
    chunks = windows(y, sr, window)
    inputs = extractor([c for c in chunks], sampling_rate=sr, return_tensors='pt').to(model.device)
    with torch.no_grad():
        probs = torch.sigmoid(model(**inputs).logits).cpu().numpy()
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


def tag_file(path, seconds=60):
    y, sr = load_audio(path, seconds=seconds, sample_rate=16000, mono=True)
    if not len(y):
        raise ValueError(f'Empty audio: {path}')
    result = summarise(*tag_samples(y, sr))
    result['seconds_tagged'] = round(len(y) / sr, 2)
    return result


def embed_audio(ys, sr=48000):
    import torch
    processor, model = clap()
    inputs = processor(audio=list(ys), sampling_rate=sr, return_tensors='pt').to(model.device)
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


def library(root, only, out, seconds, batch=16):
    """Resumable pass: tags to `out` (CSV), CLAP vectors to `<out>_clap.f32` (512 float32 per row,
    row index in `clap_row`). A changed file appends a new row; readers keep the last row per path."""
    from concurrent.futures import ThreadPoolExecutor
    done = {}
    if out.exists():
        with out.open(encoding='utf-8', newline='') as f:
            done = {r['relative_path']: r for r in csv.DictReader(f)}
    files = sorted(p for p in root.rglob('*') if p.suffix.lower() in AUDIO and p.is_file()
                   and fnmatch.fnmatch(p.relative_to(root).as_posix().lower(), (only or '*').lower()))
    todo = []
    for path in files:
        rel, stat = path.relative_to(root).as_posix(), path.stat()
        old = done.get(rel)
        if not (old and old['size_bytes'] == str(stat.st_size) and old['mtime_ns'] == str(stat.st_mtime_ns)):
            todo.append((path, rel, stat))
    print(f'{len(files)} files, {len(todo)} to tag', file=sys.stderr, flush=True)
    vectors = out.with_name(out.stem + '_clap.f32')
    new = not out.exists()
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
                    r = summarise(*tag_samples(y16, 16000))
                    vec = embed_audio([y48[:48000 * 10]])[0].astype('<f4')
                except Exception as exc:  # one bad file must not stop a 35k-file pass
                    print(f'SKIP {rel}: {exc}', file=sys.stderr, flush=True)
                    continue
                row = v.tell() // (4 * len(vec))
                v.write(vec.tobytes())
                v.flush()
                writer.writerow({'relative_path': rel, 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                                 'seconds_tagged': round(len(y16) / 16000, 2), 'speech': r['speech'],
                                 'music': r['music'], 'silence': r['silence'],
                                 'top_labels': '; '.join(f'{k} {v_}' for k, v_ in r['labels'].items()),
                                 'timeline': ' | '.join(f"{w['start']:g}-{w['end']:g}: " +
                                                        ', '.join(f'{k} {v_}' for k, v_ in w['labels'].items())
                                                        for w in r['timeline']),
                                 'clap_row': row})
                f.flush()
            print(f'{min(b + batch, len(todo))}/{len(todo)}', file=sys.stderr, flush=True)


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
    print(f'PASS audio_tag check on {device()}: tone {list(tone_labels)[:2]}, noise {list(noise_labels)[:2]}, CLAP ranks both')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('tag')
    t.add_argument('files', nargs='*', type=Path)
    t.add_argument('--root', type=Path)
    t.add_argument('--only', help='relative-path glob under --root')
    t.add_argument('--out', type=Path, help='CSV for a resumable library pass (needs --root)')
    t.add_argument('--seconds', type=float, default=60)
    r = sub.add_parser('rank')
    r.add_argument('query')
    r.add_argument('files', nargs='+', type=Path)
    r.add_argument('--seconds', type=float, default=10)
    sub.add_parser('check')
    a = parser.parse_args()
    if a.cmd == 'check':
        return check()
    if a.cmd == 'rank':
        return print(json.dumps(rank(a.query, a.files, a.seconds), indent=1))
    if a.out:
        if not a.root:
            parser.error('--out needs --root')
        return library(a.root, a.only, a.out, a.seconds)
    files = a.files or [p for p in sorted(a.root.rglob('*')) if p.suffix.lower() in AUDIO and
                        fnmatch.fnmatch(p.relative_to(a.root).as_posix().lower(), (a.only or '*').lower())]
    print(json.dumps({str(p): tag_file(p, a.seconds) for p in files}, indent=1))


if __name__ == '__main__':
    main()
