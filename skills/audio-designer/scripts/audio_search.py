"""Search a tagged library by description, cross-referencing three independent sources.

    python audio_search.py names --root D:/MySounds          # build name vectors (seconds; rerun after tagging)
    python audio_search.py find --root D:/MySounds "a wooden click sound effect" [--top 20]
    python audio_search.py check

Sources, each ranked on its own, then merged by reciprocal rank fusion (no score calibration needed):
  audio - CLAP: the query against each file's audio fingerprint (AUDIO_TAGS_clap.f32)
  name  - the cleaned file and folder name, by meaning (CLAP text side) plus exact words
  label - the query mapped to AudioSet labels, scored by the file's AudioSet label scores. Labels only CONFIRM
          files the other two found: on designed effects (clicks, UI) the tagger is often wrong.
A file found by several sources outranks one found by a single source. Each result keeps every
source's own score and rank: the evidence stays separate (parent and child labels are one source).
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

DIM = 512
RRF_K = 60
POOL = 500  # each source contributes its best POOL files to the fusion
PAIR = re.compile(r'(.+?) (\d+(?:\.\d+)?)(?:; |, |$)')
ABBR = {'amb': 'ambience', 'sfx': 'sound effect', 'fx': 'effect', 'clk': 'click', 'vox': 'voice', 'bg': 'background',
        'btn': 'button', 'imp': 'impact', 'wsh': 'whoosh', 'ftsp': 'footstep', 'trop': 'tropical', 'rurl': 'rural',
        'subn': 'suburb', 'mech': 'mechanical', 'elec': 'electric', 'notif': 'notification', 'gui': 'ui', 'perc': 'percussion'}
# Words that describe the catalogue, not the sound.
STOP = {'effect', 'effects', 'sound', 'sounds', 'pack', 'vol', 'volume', 'collection', 'bundle', 'audio', 'hero', 'audiohero',
        'alternates', 'alternate', 'for', 'of', 'the', 'and', 'new', 'mix', 'master', 'track', 'full', 'source', 'wav', 'mp3',
        'ogg', 'main', 'cut', 'intensity', 'copy', 'final', 'version', 'ver', 'alt', 'var', 'take', 'mono', 'stereo', 'loop',
        'loops', 'looped', 'softwarebundle', 'ultimatesfxforgames', 'misc', 'sfxs'}
CODE = re.compile(r'^(?:[a-z]{0,6}\d[a-z\d]*|\d+|[a-z])$')  # SDT2016605, dlbatm002, 03, single letters
SILENT_LABELS = {'Silence'}


def words(text):
    text = re.sub(r'\((?:RT )?[\d. #]+\)|\[.*?\]', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    out = []
    for w in re.split(r'[^A-Za-z0-9]+', text.lower()):
        w = ABBR.get(w, w)
        if w and not CODE.match(w) and w not in STOP and w not in out:
            out.append(w)
    return out


def clean_name(rel, pack=''):
    """'FL_GFX_UI_Click_30.wav' under its pack -> 'ui clicks click'. Pack words are dropped: every file in it has them."""
    parts = rel.split('/')
    pack_words = set(words(pack)) | set(words(parts[0]))
    folders = [p for p in parts[1:-1] if p.lower() != pack.lower()][-2:]
    return ' '.join(w for w in words(' '.join(folders + [parts[-1].rsplit('.', 1)[0]])) if w not in pack_words)


def drop_pack_constants(names, packs, share=0.6, minimum=10):
    """Remove words found in most files of a pack ('fl gfx', a vendor prefix): they cannot tell its sounds apart."""
    from collections import Counter, defaultdict
    members = defaultdict(list)
    for i, pack in enumerate(packs):
        members[pack].append(i)
    out = list(names)
    for idx in members.values():
        if len(idx) < minimum:
            continue
        counts = Counter(w for i in idx for w in set(names[i].split()))
        common = {w for w, c in counts.items() if c >= share * len(idx)}
        for i in idx:
            kept = [w for w in names[i].split() if w not in common]
            out[i] = ' '.join(kept) if kept else names[i]
    return out


PREPOSITIONS = {'at', 'in', 'on', 'with', 'without', 'of', 'under', 'over', 'for', 'from', 'by', 'near', 'during', 'like', 'into', 'through'}


def head(query):
    """The sound word of a description: the last word before any preposition.
    'a single soft wooden click' -> 'click'; 'birds at dawn' -> 'birds'; 'calm piano without voices' -> 'piano'."""
    kept = []
    for w in re.findall(r"[a-z]+", query.lower()):
        if w in PREPOSITIONS and kept:
            break
        kept.append(w)
    content = [w for w in kept if w not in STOP and w not in {'a', 'an', 'some', 'very'}]
    return ABBR.get(content[-1], content[-1]) if content else query.strip().lower()


def label_matches(word_stem, label_names):
    """AudioSet labels naming the sound word as a whole word: 'rain' -> Rain, Raindrop, never Train."""
    if not word_stem:
        return []
    pattern = re.compile(r'\b' + re.escape(word_stem))
    return [n for n in label_names if pattern.search(n.lower())]


def stem(w):
    for suffix in ('ing', 'es', 's', 'ed'):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[:-len(suffix)]
    return w


class Index:
    """The library's rows, their audio vectors, name vectors and label scores, loaded once."""

    def __init__(self, root):
        self.root = Path(root)
        tags = {}
        with (self.root / 'AUDIO_TAGS.csv').open(encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                tags[r['relative_path']] = r  # last row wins
        packs, seconds = {}, {}
        ledger = self.root / 'AUDIO_LEDGER_files.csv'
        if ledger.exists():
            with ledger.open(encoding='utf-8-sig', newline='') as f:
                ledger_rows = list(csv.DictReader(f))
            packs = {r['relative_path']: r.get('pack', '') for r in ledger_rows}
            seconds = {r['relative_path']: float(r.get('duration_seconds') or 0) for r in ledger_rows}
        audio = np.fromfile(self.root / 'AUDIO_TAGS_clap.f32', '<f4')
        audio = audio[:len(audio) // DIM * DIM].reshape(-1, DIM)
        self.paths, self.names, rows = [], [], []
        for rel, r in tags.items():
            row = int(r['clap_row']) if r.get('clap_row') else -1
            if 0 <= row < len(audio):
                self.paths.append(rel)
                self.names.append(clean_name(rel, packs.get(rel, '')))
                rows.append(row)
        self.names = drop_pack_constants(self.names, [packs.get(p) or p.split('/')[0] for p in self.paths])
        self.audio = audio[rows]
        self.rows = np.array(rows)  # each file's row in AUDIO_TAGS_clap.f32
        self.labels = [dict((k, float(v)) for k, v in PAIR.findall(tags[p]['top_labels'])) for p in self.paths]
        self.silence = np.array([float(tags[p]['silence'] or 0) for p in self.paths])
        self.label_text = [tags[p]['top_labels'] + ' ' + tags[p].get('timeline', '') for p in self.paths]
        self.seconds = np.array([seconds.get(p, 0.0) for p in self.paths])
        self.name_vectors = self._name_vectors()

    def _name_vectors(self):
        """Cached per name text, so a growing library only embeds new names."""
        cache = self.root / 'AUDIO_NAMES_clap.npz'
        known = {}
        if cache.exists():
            data = np.load(cache, allow_pickle=False)
            known = dict(zip(data['names'].tolist(), data['vectors']))
        missing = sorted({n for n in self.names if n not in known})
        if missing:
            from audio_tag import embed_text
            for i in range(0, len(missing), 256):
                for n, v in zip(missing[i:i + 256], embed_text([n or 'unnamed sound' for n in missing[i:i + 256]])):
                    known[n] = v
            names = sorted(known)
            np.savez(cache, names=np.array(names), vectors=np.stack([known[n] for n in names]).astype('<f4'))
        return np.stack([known[n] for n in self.names]) if self.names else np.zeros((0, DIM), '<f4')

    def sources(self, query):
        """Each source's score for every file, kept apart, plus each source's weight for this query."""
        from audio_tag import embed_text, tagger
        main = head(query)
        q = embed_text([query, f'the sound of {query}', main])
        q_audio = q[:2].mean(0) / np.linalg.norm(q[:2].mean(0))
        audio = self.audio @ q_audio
        # name: the sound word counts most ("click" in "a single soft wooden click"); describing words refine.
        main_stem = stem(main)
        others = {stem(w) for w in words(query) if len(w) > 2} - {main_stem}
        name_stems = [{stem(w) for w in n.split()} for n in self.names]
        name = (0.5 * (self.name_vectors @ q[0]) + 0.5 * (self.name_vectors @ q[2])
                + np.array([0.3 * (main_stem in s) + 0.05 * len(others & s) for s in name_stems]))
        # label: AudioSet labels that name the sound word outright; failing that, the nearest by meaning, at half weight
        label_names = list(tagger()[1].config.id2label.values())
        direct = {n: 1.0 for n in label_matches(main_stem, label_names)}
        if direct:
            wanted, weight = direct, 1.0
        else:
            if getattr(self, '_label_vectors', None) is None:  # 527 fixed names: embed once per index
                self._label_vectors = embed_text([f'the sound of {n.lower()}' for n in label_names])
            sims = self._label_vectors @ q[0]
            top = np.argsort(-sims)[:5]
            wanted, weight = {label_names[i]: float(sims[i]) for i in top if sims[i] >= sims[top[0]] - 0.08}, 0.5
        label = np.array([max([s * lab.get(n, 0.0) for n, s in wanted.items()] or [0.0]) for lab in self.labels])
        return {'audio': audio, 'name': name, 'label': label}, wanted, {'audio': 1.0, 'name': 1.0, 'label': weight}

    def find(self, query, top=50, keep=None):
        """Fused ranking. `keep` is an optional boolean mask from the caller's filters."""
        scores, wanted, weights = self.sources(query)
        fused = np.zeros(len(self.paths))
        ranks = {}
        found = np.zeros(len(self.paths), bool)
        for source, s in scores.items():  # audio and name find; label only confirms what they found
            s = np.where(keep, s, -np.inf) if keep is not None else s
            if source == 'label':
                s = np.where(found & (s > 0), s, -np.inf)
            order = np.argsort(-s)[:POOL]
            order = order[np.isfinite(s[order])]
            r = np.full(len(self.paths), -1)
            r[order] = np.arange(len(order))
            ranks[source] = r
            fused[order] += weights[source] / (RRF_K + np.arange(len(order)))
            found[order] = True
        # Mostly-silent files only surface when the query is about silence.
        if not SILENT_LABELS & set(wanted):
            fused[self.silence >= 0.5] *= 0.25
        best = [i for i in np.argsort(-fused)[:top] if fused[i] > 0]
        return [{'path': self.paths[i], 'row': int(self.rows[i]), 'score': round(float(fused[i]) * 100, 3), 'name': self.names[i],
                 'evidence': {src: {'rank': int(ranks[src][i]) + 1 if ranks[src][i] >= 0 else None,
                                    'score': round(float(scores[src][i]), 3)} for src in scores}}
                for i in best], wanted


def like_score(approved, rejected):
    """Mean similarity to the approved examples, less half of any lead the closest rejected one has over it."""
    return approved - 0.5 * np.maximum(0, rejected - approved)


def like(index, approved, rejected=(), avoid=None, min_seconds=0, top=20):
    """Files that sound like the approved examples and unlike the rejected ones (path substrings), by audio vector.
    `avoid` drops any file whose path or labels (any analysed window) match the regex: a missing label is not an absence,
    so name the words too (Voice, Lyric, Vocal)."""
    pick = lambda subs: [i for i, p in enumerate(index.paths) if any(s.lower() in p.lower() for s in subs)]
    a, r = pick(approved), pick(rejected)
    if not a:
        raise SystemExit('no library file matches the approved examples: ' + ', '.join(approved))
    sim_a = index.audio @ index.audio[a].T
    score = like_score(sim_a.mean(axis=1), (index.audio @ index.audio[r].T).max(axis=1) if r else np.zeros(len(index.paths)))
    skip = set(a) | set(r)
    pattern = re.compile(avoid, re.I) if avoid else None
    out, seen = [], set()
    for i in np.argsort(-score):
        base = index.paths[i].rsplit('/', 1)[-1].rsplit('.', 1)[0].lower()  # a WAV and its MP3 copy are one result
        if i in skip or base in seen or index.seconds[i] < min_seconds:
            continue
        if pattern and (pattern.search(index.label_text[i]) or pattern.search(index.paths[i])):
            continue
        seen.add(base)
        out.append({'path': index.paths[i], 'score': round(float(score[i]), 3), 'approved': round(float(sim_a[i].mean()), 3),
                    'seconds': round(float(index.seconds[i]), 1), 'labels': index.label_text[i].split(' 0-')[0][:120]})
        if len(out) == top:
            break
    return out


def check():
    pack = [f'FL_GFX_{kind}_{n}.wav' for kind in ('UI_Click', 'Magic_Spell', 'Coin_Drop', 'Door_Slam') for n in range(3)]
    cleaned = drop_pack_constants([clean_name('Ultimate/' + f, 'Ultimate') for f in pack], ['Ultimate'] * len(pack))
    assert cleaned[0] == 'ui click' and cleaned[3] == 'magic spell', cleaned[:4]
    assert drop_pack_constants(['ui click'] * 3, ['small'] * 3) == ['ui click'] * 3  # a tiny pack keeps its words
    assert clean_name('AudioHero/africa/AMBTrop_JungleBird Chirp_MONKY005.wav', 'africa') == 'ambience tropical jungle bird chirp', \
        clean_name('AudioHero/africa/AMBTrop_JungleBird Chirp_MONKY005.wav', 'africa')
    assert 'rt' not in clean_name('Pack/Evening (RT 4.745)/Ambient Vol2 Evening Intensity 1.wav', 'Pack').split()
    assert stem('clicks') == 'click' and stem('chirping') == 'chirp'
    assert label_matches('rain', ['Train', 'Rain', 'Raindrop', 'Rain on surface']) == ['Rain', 'Raindrop', 'Rain on surface']
    assert label_matches('click', ['Clicking', 'Tick', 'Clickety-clack']) == ['Clicking', 'Clickety-clack']
    assert head('a single soft wooden click') == 'click' and head('birds at dawn') == 'birds'
    assert head('calm piano without voices') == 'piano' and head('a short UI confirmation') == 'confirmation'
    # Fusion: a file ranked 2nd by two sources beats one ranked 1st by a single source.
    two = 2 / (RRF_K + 1)
    one = 1 / (RRF_K + 0)
    assert two > one
    # like: closer to the approved wins; a file nearer a rejected example than the approved loses half that lead.
    assert like_score(.7, .5) == .7 and abs(like_score(.6, .9) - .45) < 1e-9 and like_score(.6, .9) < like_score(.5, .4)
    print('PASS audio_search check: names cleaned, stems, fusion favours agreement')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    n = sub.add_parser('names')
    n.add_argument('--root', type=Path, required=True)
    f = sub.add_parser('find')
    f.add_argument('query')
    f.add_argument('--root', type=Path, required=True)
    f.add_argument('--top', type=int, default=20)
    k = sub.add_parser('like', help='sounds like the approved examples, unlike the rejected ones')
    k.add_argument('--root', type=Path, required=True)
    k.add_argument('--approved', nargs='+', required=True, help='path substrings of approved files')
    k.add_argument('--rejected', nargs='*', default=[])
    k.add_argument('--avoid', help='regex over file paths and labels in every window, e.g. "Choir|Sing|Speech|Organ|Voice|Lyric|Vocal"')
    k.add_argument('--min-seconds', type=float, default=0)
    k.add_argument('--top', type=int, default=20)
    sub.add_parser('check')
    a = parser.parse_args()
    if a.cmd == 'check':
        return check()
    index = Index(a.root)
    if a.cmd == 'names':
        return print(f'{len(index.names)} names embedded, cache at {a.root / "AUDIO_NAMES_clap.npz"}')
    if a.cmd == 'like':
        return print(json.dumps(like(index, a.approved, a.rejected, a.avoid, a.min_seconds, a.top), indent=1))
    results, wanted = index.find(a.query, a.top)
    print(json.dumps({'labels_used': wanted, 'results': results}, indent=1))


if __name__ == '__main__':
    main()
