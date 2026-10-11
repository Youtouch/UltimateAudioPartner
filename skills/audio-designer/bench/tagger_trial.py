"""Compare AudioSet taggers on fresh, name-sampled clips: does the tagger hear what the file name says?

    AUDIO_LIBRARY=D:/MySounds python tagger_trial.py run [--per-family 40] [--models ast ced-tiny ced-base]
    python tagger_trial.py check

For each sound family, files whose NAME says the family are drawn at random from the whole tagged library
(not from the benchmark, whose clips were partly nominated with the current tagger's labels). Each tagger
scores the same first 10 seconds; a hit is a family label among the tagger's top 5. Names are a proxy, equally
fair to every tagger, and wrong for some files: compare taggers with it, do not quote it as accuracy.
Full unrounded score vectors are saved to tagger_trial_scores.npz.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))

# family -> (file-name regex, AudioSet label-name regex)
FAMILIES = {
    'click': (r'click|clk|\btick|\btap\b|knock', r'click|tick|tap\b|knock|wood block|typing|keyboard'),
    'bird': (r'bird|songbird|chirp|tweet', r'bird|chirp|tweet|owl|crow|pigeon|fowl|goose|duck'),
    'rain': (r'\brain', r'\brain'),
    'water': (r'water|stream|river|drip|creek', r'water|stream|drip|trickle|pour|splash|gurgl|ocean|waves'),
    'speech': (r'speech|talk|dialog|conversation|narrat', r'speech|conversation|narration|babble'),
    'choir': (r'choir|chant|choral', r'choir|chant|mantra|singing|vocal music'),
    'insect': (r'insect|cricket|cicada|\bbees?\b|\bfly\b', r'insect|cricket|bee, wasp|fly, housefly|mosquito|buzz'),
    'frog': (r'frog|toad', r'frog|croak'),
    'wind': (r'\bwind\b|breeze|gust', r'\bwind|rustling leaves|howl'),
    'dog': (r'\bdogs?\b|bark', r'\bdog|bark|bow-wow|howl|growl|whimper'),
    'door': (r'\bdoors?\b', r'door|slam|knock|creak'),
    'footstep': (r'footstep|\bsteps?\b|walk', r'walk, footsteps|run|shuffle'),
    'glass': (r'glass', r'glass|chink|clink|shatter'),
    'whoosh': (r'whoosh|swoosh|swish', r'whoosh|swoosh|swish'),
    'piano': (r'piano', r'piano|keyboard \(musical\)'),
}
MODELS = {'ast': 'MIT/ast-finetuned-audioset-10-10-0.4593', 'ced-tiny': 'mispeech/ced-tiny', 'ced-base': 'mispeech/ced-base'}
# CED runs model code from its Hugging Face repo: pin the reviewed revision so an update cannot swap it silently.
# None = not yet reviewed (the first run downloads it; read its .py files, then pin).
REVISIONS = {'ced-tiny': 'ace276d29dd0bb3f3517b0fa8cf300738c409019', 'ced-base': 'db3e14a8db4c21b56b165261c39649741a900e7f'}


def sample(root, per_family, seed=20261011):
    rows = {}
    with (root / 'AUDIO_TAGS.csv').open(encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            rows[r['relative_path']] = r
    paths = sorted(rows)
    rng = random.Random(seed)
    picked = {}
    for fam, (name_re, _) in FAMILIES.items():
        pat = re.compile(name_re, re.I)
        hits = [p for p in paths if pat.search(p.rsplit('/', 1)[-1])]
        picked[fam] = rng.sample(hits, min(per_family, len(hits)))
    return picked


def load_model(key):
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    name = MODELS[key]
    remote = key.startswith('ced')  # CED ships its model code in its Hugging Face repo (Apache-2.0)
    rev = REVISIONS.get(key)
    extractor = AutoFeatureExtractor.from_pretrained(name, trust_remote_code=remote, revision=rev)
    model = AutoModelForAudioClassification.from_pretrained(name, trust_remote_code=remote, revision=rev)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    return extractor, model


def score(extractor, model, y):
    import torch
    if len(y) < 16000:  # CED's first layer needs ~0.2 s; pad every model the same way so the comparison stays fair
        y = np.pad(y, (0, 16000 - len(y)))
    inputs = extractor(y, sampling_rate=16000, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits if hasattr(out, 'logits') else out[0]
    probs = torch.sigmoid(logits) if logits.min() < 0 or logits.max() > 1 else logits
    return probs[0].float().cpu().numpy()


def hit(probs, labels, label_re, k=5):
    pat = re.compile(label_re, re.I)
    return any(pat.search(labels[i]) for i in np.argsort(-probs)[:k])


def run(root, per_family, models):
    from audio_map import load_audio
    picked = sample(root, per_family)
    audio = {}
    for fam, paths in picked.items():
        for p in paths:
            try:
                audio[p] = load_audio(root / p, seconds=10, sample_rate=16000, mono=True)[0]
            except Exception as exc:
                print(f'SKIP {p}: {exc}', file=sys.stderr)
    print(f'{len(audio)} clips decoded across {len(FAMILIES)} families', flush=True)
    report, saved = {}, {}
    for key in models:
        t0 = time.time()
        extractor, model = load_model(key)
        load_s = time.time() - t0
        labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
        scores, t1 = {}, time.time()
        for p, y in audio.items():
            scores[p] = score(extractor, model, y)
        per_clip = (time.time() - t1) / max(1, len(audio))
        fams = {}
        for fam, paths in picked.items():
            got = [p for p in paths if p in scores]
            fams[fam] = round(sum(hit(scores[p], labels, FAMILIES[fam][1]) for p in got) / max(1, len(got)), 3)
        report[key] = {'families': fams, 'mean': round(float(np.mean(list(fams.values()))), 3),
                       'load_s': round(load_s, 1), 'per_clip_s': round(per_clip, 4), 'labels': len(labels)}
        saved[key] = np.stack([scores[p] for p in audio])
        saved[key + '_labels'] = np.array(labels)
        print(key, json.dumps(report[key]), flush=True)
        del model
    np.savez(HERE / 'tagger_trial_scores.npz', paths=np.array(list(audio)), **saved)
    (HERE / 'tagger_trial.json').write_text(json.dumps({'per_family': per_family, 'picked': picked, 'report': report}, indent=1),
                                            encoding='utf-8')
    return report


def check():
    labels = ['Speech', 'Bird', 'Clicking', 'Train', 'Rain', 'Music']
    probs = np.array([0.1, 0.9, 0.05, 0.8, 0.0, 0.3])
    assert hit(probs, labels, FAMILIES['bird'][1])
    assert not hit(probs, labels, FAMILIES['rain'][1], k=3), 'Train must not count as rain'
    assert hit(probs, labels, FAMILIES['click'][1], k=6) and not hit(probs, labels, FAMILIES['click'][1], k=4)
    assert re.search(FAMILIES['click'][0], 'FL_GFX_UI_Click_30.wav', re.I) and not re.search(FAMILIES['click'][0], 'Tapestry.wav', re.I)
    print('PASS tagger_trial check: family label hits, top-k cut, Train is not rain, name patterns')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--per-family', type=int, default=40)
    r.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    sub.add_parser('check')
    a = parser.parse_args()
    if a.cmd == 'check':
        return check()
    run(Path(os.environ.get('AUDIO_LIBRARY', '.')), a.per_family, a.models)


if __name__ == '__main__':
    main()
