"""Short game sound effects: a search benchmark, and an experiment on how the label model hears them.

    AUDIO_LIBRARY=D:/MySounds python short_sfx.py search [--out short_sfx_search.json]
    AUDIO_LIBRARY=D:/MySounds python short_sfx.py labels [--per-query 120] [--out short_sfx_labels.json]
    python short_sfx.py check

PROXY truth: a file is relevant to a query when its FILE NAME says so (QUERIES below), restricted to files
under 1.5 s. Names are wrong for some files and flatter name-based search: compare methods with it, never
quote it as accuracy. Packs are split DEV/TEST by a fixed hash (tagging/label_rules.pack_splits), so no pack
contributes to both.

`search` ranks every short file per query by name only, audio only, labels only and fused (what the browser
runs), and reports AP and P@10 per split. `labels` asks how the AudioSet label model hears short sounds: the
same name-matched files scored with silence padding to 1 s (what tagging does today), 2 s or 10 s, with the
sound at the start or the centre, from the raw file or from the sound with its silences trimmed. A hit is one
of the query's AudioSet labels in the model's top 5. The best condition on DEV is compared with today's on TEST
by an exact paired McNemar test; a change is adopted only if TEST favours it at p < 0.05.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
sys.path.insert(0, str(HERE.parent / 'tagging'))

MAX_SECONDS = 1.5
# query id: (query, file-name regex (PROXY truth), name exclusion regex, AudioSet labels that count as hearing it)
QUERIES = {
    'footstep_gravel': ('footstep on gravel', r'(foot ?steps?|steps?|walk|run).*gravel|gravel.*(foot ?steps?|steps?|walk|run)', '',
                        r'^Walk, footsteps$|^Run$|^Shuffle$|^Crunch$'),
    'sword_swing': ('sword swing', r'sword.*(swing|swish|whoosh|slash|swipe)|(swing|swish|whoosh|slash|swipe).*sword', '',
                    r'^Whoosh, swoosh, swish$'),
    'coin_pickup': ('coin pickup', r'\bcoins?\b|coin', r'coinc', r'^Coin \(dropping\)$|^Cash register$|^Jingle, tinkle$|^Chink, clink$'),
    'ui_click': ('UI button click', r'(\bui\b|button|menu|gui|interface).*(click|press|tap)|(click|press).*(button|\bui\b|menu)', '',
                 r'^Clicking$|^Tick$|^Tap$|^Computer keyboard$|^Typing$'),
    'gunshot': ('gunshot', r'gun ?shot|pistol|rifle|shotgun|gunfire|\bgun\b', r'reload|shell|casing|empty|holster|cock',
                r'^Gunshot, gunfire$|^Machine gun$|^Fusillade$|^Artillery fire$|^Cap gun$'),
    'laser': ('laser shot', r'laser|blaster|phaser', '', r'^Beep, bleep$|^Sine wave$|^Chirp tone$|^Zing$|^Synthesizer$'),
    'jump': ('jump', r'\bjump', '', r'^Boing$|^Whoosh, swoosh, swish$'),
    'punch': ('punch impact', r'punch', r'puncher|punch ?card', r'^Slap, smack$|^Whack, thwack$|^Thump, thud$|^Smash, crash$'),
    'glass_break': ('glass breaking', r'glass.*(break|shatter|smash|crash)|(break|shatter|smash).*glass|shatter', '',
                    r'^Shatter$|^Glass$|^Breaking$|^Chink, clink$'),
    'explosion': ('explosion', r'explo|blast|\bboom\b|detonat', r'blaster', r'^Explosion$|^Boom$|^Burst, pop$|^Eruption$'),
    'magic_spell': ('magic spell', r'magic|spell|enchant|sorcer|wizard', '', r'^Chime$|^Wind chime$|^Jingle, tinkle$|^Bell$|^Synthesizer$'),
    'notification': ('notification sound', r'notif|alert|\bping\b|message', '', r'^Beep, bleep$|^Ding$|^Chime$|^Ding-dong$|^Bell$'),
}
METHODS = ('name', 'audio', 'label', 'fused')
CONDITIONS = [(trim, place, pad) for trim in ('raw', 'trimmed') for place in ('start', 'centre') for pad in (1, 2, 10)]
BASELINE = ('raw', 'start', 1)


def proxy(query_id, path):
    _, name_re, exclude, _ = QUERIES[query_id]
    name = path.rsplit('/', 1)[-1]
    return bool(re.search(name_re, name, re.I)) and not (exclude and re.search(exclude, name, re.I))


def universe(index):
    """Short files with a fingerprint, their DEV/TEST split by pack."""
    import label_rules
    keep = (index.seconds > 0) & (index.seconds < MAX_SECONDS)
    packs = [p.split('/')[0] for p in index.paths]
    split = label_rules.pack_splits(packs)
    return keep, split


def search(root):
    from audio_search import Index
    from bench_eval import retrieval_metrics
    index = Index(root)
    keep, split = universe(index)
    report = {'truth': 'PROXY - file names, files under 1.5 s', 'universe': {s: int((keep & (split == s)).sum()) for s in ('dev', 'test')},
              'queries': {}}
    for qid, (query, *_rest) in QUERIES.items():
        scores, _, _ = index.sources(query)
        per = {}
        for s in ('dev', 'test'):
            mask = keep & (split == s)
            positives = {index.paths[i] for i in np.flatnonzero(mask) if proxy(qid, index.paths[i])}
            per[s] = {'relevant': len(positives)}
            for method in METHODS:
                if method == 'fused':
                    ranking = [r['path'] for r in index.find(query, top=int(mask.sum()), keep=mask)[0]]
                else:
                    sc = np.where(mask, scores[method], -np.inf)
                    order = np.argsort(-sc, kind='stable')[:int(mask.sum())]
                    ranking = [index.paths[i] for i in order if np.isfinite(sc[i]) and (method != 'label' or sc[i] > 0)]
                m = retrieval_metrics(ranking, positives, int(mask.sum()))
                per[s][method] = {'AP': m['AP'], 'P@10': m['P@10']}
        report['queries'][qid] = per
        print(qid, json.dumps({s: {m: round(per[s][m]['AP'] or 0, 3) for m in METHODS} for s in per}), flush=True)
    report['macro'] = {s: {m: {k: _mean([q[s][m][k] for q in report['queries'].values() if q[s]['relevant']])
                               for k in ('AP', 'P@10')} for m in METHODS} for s in ('dev', 'test')}
    return report


def _mean(values):
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 4) if values else None


def prepare(y, sr, trim, place, pad):
    """One condition's model input from a short file's 16 kHz mono samples."""
    import audio_describe
    if trim == 'trimmed':
        points = audio_describe.trim_points(y, float(np.abs(y).max()) if len(y) else 0.0)
        if points:
            y = y[points[0]:points[1] + 1]
    size = int(pad * sr)
    if len(y) >= size:
        return y
    left = (size - len(y)) // 2 if place == 'centre' else 0
    return np.pad(y, (left, size - len(y) - left))


def hits_for(probs, label_names, label_re, k=5):
    pattern = re.compile(label_re)
    return [any(pattern.search(label_names[j]) for j in np.argsort(-row)[:k]) for row in probs]


def mcnemar(new, old):
    """Exact two-sided McNemar on paired booleans: (files only the new condition hears, only the old, p)."""
    from scipy.stats import binomtest
    new, old = np.asarray(new), np.asarray(old)
    b, c = int((new & ~old).sum()), int((~new & old).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)


def labels(root, per_query, seed=20261012):
    import torch
    from audio_map import load_audio
    from audio_search import Index
    from audio_tag import tagger, TAGGER
    index = Index(root)
    keep, split = universe(index)
    rng = np.random.default_rng(seed)
    files = []  # (query id, path, split)
    for qid in QUERIES:
        for s in ('dev', 'test'):
            found = [index.paths[i] for i in np.flatnonzero(keep & (split == s)) if proxy(qid, index.paths[i])]
            for p in sorted(rng.choice(found, min(per_query, len(found)), replace=False)) if found else []:
                files.append((qid, p, s))
    print(f'{len(files)} name-matched short files', flush=True)
    audio = {}
    for _, p, _ in files:
        if p not in audio:
            try:
                audio[p] = load_audio(Path(root) / p, seconds=MAX_SECONDS, sample_rate=16000, mono=True)[0]
            except Exception as exc:
                print(f'SKIP {p}: {exc}', file=sys.stderr)
    files = [f for f in files if f[1] in audio and len(audio[f[1]])]
    extractor, model = tagger()
    names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    hits, seconds = {}, {}
    for condition in CONDITIONS:
        start = time.perf_counter()
        probs = []
        for _, p, _ in files:  # one file per pass, as tagging does: batching would pad every file to the longest
            inputs = extractor([prepare(audio[p], 16000, *condition)], sampling_rate=16000, return_tensors='pt').to(model.device)
            with torch.no_grad():
                out = model(**inputs).logits
                probs.append((out if TAGGER.startswith('ced') else torch.sigmoid(out)).float().cpu().numpy()[0])
        probs = np.stack(probs)
        key = '/'.join(map(str, condition))
        hits[key] = np.array([hits_for(probs[i:i + 1], names, QUERIES[q][3])[0] for i, (q, _, _) in enumerate(files)])
        seconds[key] = round((time.perf_counter() - start) / len(files), 4)
        print(key, {s: int(hits[key][[f[2] == s for f in files]].sum()) for s in ('dev', 'test')}, flush=True)
    return decide(files, hits, seconds)


def decide(files, hits, seconds):
    """Pick on DEV, test once on TEST against the baseline."""
    dev = np.array([f[2] == 'dev' for f in files])
    base = '/'.join(map(str, BASELINE))
    dev_hits = {k: int(v[dev].sum()) for k, v in hits.items()}
    best = max(hits, key=lambda k: (dev_hits[k], k == base))  # ties keep today's condition
    b, c, p = mcnemar(hits[best][~dev], hits[base][~dev])
    adopt = best != base and b > c and p < 0.05
    per_query = {q: {s: {k: int(v[[f[0] == q and f[2] == s for f in files]].sum()) for k, v in hits.items()}
                     for s in ('dev', 'test')} for q in QUERIES}
    files_per = {q: {s: sum(f[0] == q and f[2] == s for f in files) for s in ('dev', 'test')} for q in QUERIES}
    return {'truth': 'PROXY - file names; a hit is a query label in the top 5', 'files': {'dev': int(dev.sum()), 'test': int((~dev).sum())},
            'dev_hits': dev_hits, 'test_hits': {k: int(v[~dev].sum()) for k, v in hits.items()},
            'chosen_on_dev': best, 'baseline': base, 'test_mcnemar': {'new_only': b, 'old_only': c, 'p': p},
            'adopt': adopt, 'seconds_per_file': seconds, 'per_query': per_query, 'files_per_query': files_per}


def check():
    assert proxy('footstep_gravel', 'Pack/Footsteps_Gravel_03.wav') and proxy('footstep_gravel', 'P/gravel run 2.wav')
    assert not proxy('footstep_gravel', 'P/Footsteps_Wood_01.wav')
    assert proxy('gunshot', 'P/Pistol_Shot_01.wav') and not proxy('gunshot', 'P/Pistol_Reload_01.wav')
    assert not proxy('coin_pickup', 'P/Coincidence.wav') and proxy('ui_click', 'P/UI_Button_Click_2.wav')
    y = np.concatenate([np.zeros(800), np.ones(1600) * 0.5, np.zeros(800)]).astype('float32')
    assert len(prepare(y, 16000, 'raw', 'start', 1)) == 16000 and prepare(y, 16000, 'raw', 'start', 1)[800] == 0.5
    centred = prepare(y, 16000, 'trimmed', 'centre', 2)
    assert len(centred) == 32000 and np.flatnonzero(centred)[0] == (32000 - 1600) // 2, 'trimmed sound sits in the middle'
    assert len(prepare(np.ones(40000, 'float32'), 16000, 'raw', 'start', 2)) == 40000, 'longer than the pad: unchanged'
    assert hits_for(np.array([[.1, .9, .2]]), ['Speech', 'Gunshot, gunfire', 'Music'], QUERIES['gunshot'][3], k=1) == [True]
    # Decision rule: picked on DEV, adopted only when TEST favours it significantly; a tie keeps the baseline.
    files = [('q', f'p{i}', 'dev' if i < 40 else 'test') for i in range(80)]
    base, better = np.zeros(80, bool), np.zeros(80, bool)
    better[:20] = better[40:60] = True
    d = decide(files, {'raw/start/1': base, 'raw/start/2': better}, {})
    assert d['chosen_on_dev'] == 'raw/start/2' and d['adopt'] and d['test_mcnemar']['new_only'] == 20
    d = decide(files, {'raw/start/1': base, 'raw/start/2': base.copy()}, {})
    assert d['chosen_on_dev'] == 'raw/start/1' and not d['adopt']
    worse_on_test = better.copy()
    worse_on_test[40:] = False
    assert not decide(files, {'raw/start/1': base, 'raw/start/2': worse_on_test}, {})['adopt'], 'DEV gain alone is not enough'
    print('PASS short_sfx check: proxy names, padding/centring/trimming, top-k hits, DEV pick then TEST McNemar')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('search')
    s.add_argument('--out', type=Path, default=HERE / 'short_sfx_search.json')
    lab = sub.add_parser('labels')
    lab.add_argument('--per-query', type=int, default=120)
    lab.add_argument('--out', type=Path, default=HERE / 'short_sfx_labels.json')
    sub.add_parser('check')
    a = parser.parse_args()
    if a.cmd == 'check':
        return check()
    root = Path(os.environ.get('AUDIO_LIBRARY', '.'))
    report = search(root) if a.cmd == 'search' else labels(root, a.per_query)
    a.out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report.get('macro') or {k: report[k] for k in ('chosen_on_dev', 'test_mcnemar', 'adopt')}, indent=1))


if __name__ == '__main__':
    main()
