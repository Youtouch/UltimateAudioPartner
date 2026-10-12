"""Stratified, pack-grouped review pilot. No model or audio writes."""
import os
import argparse
import csv
import hashlib
import io
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
from library_metadata import read_metadata, tag_metadata, relative_path
ROOT = Path(os.environ.get('AUDIO_LIBRARY', '.'))  # the library folder holding the ledger, tags and vectors
FIELDS = ['relative_path','group_id','split','stratum','duration_seconds','sources',
          'foreground_or_background','uncertain','notes','reviewed_by']


def check():
    assert family('Pack/MP3/Track Cut 30.mp3') == family('Pack/WAV/Track Intensity 1.wav')
    assert family('Pack/Hit A.wav') == family('Pack/Hit B.ogg')
    assert pack_key('Epic Music Pack Vol. 2 (1)') == pack_key('Epic Music Pack Vol. 2')
    labels = parse_labels('Other, label 0.62; Bird 0.91')
    assert labels == {'Other, label': .62, 'Bird': .91}
    rows, meta = latest_csv_bytes(b'relative_path,clap_row\na.wav,1\na.wav,2\nb.wav,')
    assert rows['a.wav']['clap_row'] == '2' and 'b.wav' not in rows
    assert meta['duplicates'] == 1 and meta['ignored_tail_bytes'] == 6
    fixtures = [{'relative_path':f'P{i}/Click {j}.wav','pack':f'P{i}',
                 'duration_seconds':'.2','bucket':'sfx','top_labels':'Click 0.7',
                 'silence':'0','music':'0','speech':'0','title_tag':''}
                for i in range(12) for j in range(10)]
    chosen, audit, stats = select(fixtures, seed=41, per_stratum=3, pack_cap=2)
    assert len(chosen) > 0
    assert max(Counter(x['group_id'] for x in chosen).values()) <= 2
    assert len({x['relative_path'] for x in chosen}) == len(chosen)
    by_group = defaultdict(set)
    for x in chosen: by_group[x['group_id']].add(x['split'])
    assert all(len(s) == 1 for s in by_group.values())
    assert all(not x[k] for x in chosen for k in FIELDS[5:])
    assert all(0 <= float(x['duration_seconds']) < 1 for x in chosen if x['stratum']=='short_one_shots')
    ledger_fallback_check()
    print('PASS bench_select check: variants, duplicate-last, append tail, pack cap, split isolation, blank truth, ledger fallback')


# Patterns only nominate candidates; they are never written into truth columns.
RULES = {
 'short_one_shots': (r'click|\btap\b|tick|button|ui[_ ]', r'knock|impact|beep|chirp'),
 'ui_game': (r'ui[_ /]|interface|button|confirm|notification|select|menu', r'click|beep|chime|coin|chirp'),
 'birds': (r'bird|chirp|tweet|owl|crow|parrot|seagull|rooster', r'insect|cricket|frog|squeak|whistle'),
 'insects': (r'insect|cricket|cicada|mosquito|bee\b|buzz', r'bird|frog|rain|static|electri'),
 'frogs': (r'frog|croak|toad', r'bird|cricket|insect|water|swamp'),
 'water_rain': (r'water|rain|stream|river|ocean|wave|drip|splash', r'wind|static|sizzle|insect'),
 'wind': (r'wind|breeze|gust', r'whoosh|noise|rain|breath'),
 'speech': (r'speech|voicepack|talk|dialog|narrat|spoken|conversation', r'sing|choir|chant|crowd|organ'),
 'sung_choral': (r'choir|chant|singing|vocal music|sung|choral|mantra', r'organ|pad|synth|ethereal|angels|halcyon'),
 'organ_pads': (r'organ|pad|synthesizer|drone', r'choir|chant|singing|mantra|vocal'),
 'calm_instrumental': (r'piano|calm|ambient music|meditation|relax|healing|lullaby|daydream|wind spirits', r'choir|chant|singing|speech|organ'),
 'quiet_silence': (r'silence|room.?tone|quiet|silent', r'rain|wind|noise|drone|ambient'),
 'long_ambience': (r'ambient|ambience|forest|jungle|night|room|wind|rain|ocean|drone', r'music|choir|speech|impact'),
 'impacts_whooshes': (r'impact|whoosh|swoosh|swish|hit|slam|bang|explosion', r'wind|tap|click|door|thunder')
}


def pack_key(pack):
    return re.sub(r'\s*\(\d+\)$','',pack.replace('\\','/').casefold()).strip()


def family(path):
    p = Path(path.replace('\\','/'))
    parts = [x.casefold() for x in p.parts[:-1] if not re.fullmatch(r'(mp3|wav|ogg|flac|aiff?|\d+\.?\d*\s*khz.*)',x,re.I)]
    stem = re.sub(r'\b(cut|intensity|version|variation|variant|take|loop|edit|alt)\s*[_ -]*\d*\b', '', p.stem, flags=re.I)
    stem = re.sub(r'[_ -]+(?:\d+|[a-z])$','',stem,flags=re.I)
    stem = re.sub(r'[^a-z0-9]+',' ',stem.casefold()).strip()
    return '/'.join(parts+[stem])


def parse_labels(value):
    result = {}
    for piece in value.split(';'):
        m = re.fullmatch(r'\s*(.*?)\s+([0-9]*\.?[0-9]+)\s*',piece)
        if m: result[m[1]] = float(m[2])
    return result


def latest_csv_bytes(raw):
    end = raw.rfind(b'\n')+1
    complete = raw[:end]
    reader = csv.DictReader(io.StringIO(complete.decode('utf-8-sig')), strict=True)
    latest, total = {}, 0
    for row in reader:
        if not row.get('relative_path') or None in row or any(v is None for v in row.values()):
            raise ValueError('Malformed complete CSV row')
        total += 1
        latest[row['relative_path'].replace('\\','/')] = row
    return latest, dict(rows=total,duplicates=total-len(latest),ignored_tail_bytes=len(raw)-end,
                        sha256=hashlib.sha256(complete).hexdigest(),bytes_read=len(raw))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f,fields or list(rows[0]),extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def candidate(row, stratum, role):
    duration = float(row['duration_seconds'])
    if stratum == 'short_one_shots' and not 0 < duration < 1: return False
    if stratum == 'long_ambience' and duration < 60: return False
    # Specific name + exported labels are candidate evidence only.
    text = ' '.join([row['relative_path'],row.get('title_tag',''),row.get('top_labels','')]).casefold()
    pattern = RULES[stratum][role == 'hard_negative']
    if stratum == 'quiet_silence' and role == 'positive_candidate':
        return bool(re.search(pattern,text) or float(row.get('silence') or 0) >= .5)
    if stratum == 'calm_instrumental' and role == 'positive_candidate':
        bucket = row.get('bucket')
        # A fresh metadata ledger has no editorial bucket. Music scores nominate candidates only.
        if bucket:
            musical = bucket == 'music'
        else:
            try:
                musical = float(row.get('music') or parse_labels(row.get('top_labels', '')).get('Music', 0)) >= .5
            except (TypeError, ValueError):
                musical = False
        return musical and bool(re.search(pattern,text))
    return bool(re.search(pattern,text))


def assign_splits(chosen, seed):
    groups = sorted({r['group_id'] for r in chosen})
    ndev = max(1, min(len(groups)-1, round(.6*len(groups)))) if len(groups)>1 else 1
    by = defaultdict(Counter)
    for r in chosen: by[r['group_id']][r['stratum']] += 1
    total = Counter(r['stratum'] for r in chosen)
    rng, best = random.Random(seed), None
    for _ in range(4000):
        shuffled = groups[:]; rng.shuffle(shuffled)
        dev = set(shuffled[:ndev]); counts = Counter()
        for g in dev: counts.update(by[g])
        cost = sum(((counts[s]-.6*n)/max(n,1))**2 for s,n in total.items())
        cost += 4*((sum(counts.values())-.6*len(chosen))/max(len(chosen),1))**2
        if best is None or cost < best[0]: best = cost, dev
    for r in chosen: r['split'] = 'dev' if r['group_id'] in best[1] else 'test'


def select(rows, seed=20261010, per_stratum=18, pack_cap=8):
    pools = {}
    rng = random.Random(seed)
    for s in RULES:
        for role in ('positive_candidate','hard_negative'):
            pool = [r for r in rows if candidate(r,s,role)]
            rng.shuffle(pool); pools[s,role] = pool
    # Scarce strata get first choice. Fill positives before confusable negatives.
    jobs = [(s,role,per_stratum-per_stratum//3 if role=='positive_candidate' else per_stratum//3)
            for role in ('positive_candidate','hard_negative')
            for s in sorted(RULES,key=lambda s:len(pools[s,role]))]
    chosen, audit, used, families, packs, shortfalls = [], [], set(), set(), Counter(), []
    for s,role,target in jobs:
        n = 0
        for r in pools[s,role]:
            p, pk, fam = r['relative_path'], pack_key(r['pack']), family(r['relative_path'])
            if p in used or fam in families or packs[pk] >= pack_cap: continue
            g = 'pack_'+hashlib.sha256(pk.encode()).hexdigest()[:16]
            out = dict.fromkeys(FIELDS,'')
            out.update(relative_path=p,group_id=g,stratum=s,duration_seconds=r['duration_seconds'])
            chosen.append(out)
            audit.append(dict(relative_path=p,stratum=s,candidate_role=role,pack=r['pack'],
                              variant_family=fam,top_labels=r.get('top_labels',''),
                              evidence='name/ledger/exported model labels; listening pending'))
            used.add(p); families.add(fam); packs[pk]+=1; n+=1
            if n >= target: break
        if n < target: shortfalls.append(dict(stratum=s,role=role,wanted=target,selected=n))
    assign_splits(chosen,seed)
    return chosen,audit,dict(shortfalls=shortfalls,eligible_candidates={s+' / '+r:len(p) for (s,r),p in pools.items()})


def build(args):
    if (HERE/'benchmark.csv').exists():
        raise ValueError('benchmark.csv already exists; preserve reviews. Use a NEW output folder with --out.')
    size = args.tags.stat().st_size
    with args.tags.open('rb') as f: raw = f.read(size)
    tags, metadata = latest_csv_bytes(raw)
    vector_bytes = args.vectors.stat().st_size
    vector_rows = vector_bytes//2048
    if args.ledger is None:
        ledger_path = args.tags.parent / 'AUDIO_LEDGER_files.csv'
        ledger = list(read_metadata(args.tags.parent, tags).values())
    else:
        # An explicit custom ledger keeps its own row subset and errors if that file is missing.
        ledger_path = args.ledger
        fallback = tag_metadata(tags)
        ledger = []
        for row in read_csv(ledger_path):
            rel = relative_path(row['relative_path'])
            ledger.append({**fallback.get(rel, {}), **row, 'relative_path': rel})
    eligible, rejected = [], Counter()
    for r in ledger:
        t = tags.get(r['relative_path'])
        if not t: rejected['not_tagged']+=1; continue
        try:
            index = int(t['clap_row']); duration = float(r['duration_seconds'])
            if index < 0 or index >= vector_rows or not duration > 0: raise ValueError()
        except (ValueError,TypeError): rejected['invalid_vector_or_duration']+=1; continue
        eligible.append(dict(r,**{k:v for k,v in t.items() if k != 'relative_path'}))
    chosen,audit,stats = select(eligible,args.seed,args.per_stratum,args.pack_cap)
    chosen.sort(key=lambda r:(r['split'],r['stratum'],r['relative_path']))
    write_csv(HERE/'benchmark.csv',chosen,FIELDS)
    write_csv(HERE/'selection_audit.csv',audit,
              ['relative_path','stratum','candidate_role','pack','variant_family','top_labels','evidence'])
    snapshot_fields = list(eligible[0]) if eligible else list(dict.fromkeys(
        ['relative_path','pack','duration_seconds'] + list(next(iter(tags.values()), {}))))
    write_csv(HERE/'snapshot.csv',eligible,snapshot_fields)
    spans = [dict(relative_path=r['relative_path'],start_seconds=0,
                  end_seconds=min(10.,float(r['duration_seconds'])),
                  scope='fixed opening excerpt; whole file NOT reviewed') for r in chosen]
    write_csv(HERE/'review_spans.csv',spans,['relative_path','start_seconds','end_seconds','scope'])
    metadata.update(created_utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,
                    ledger=str(ledger_path),ledger_missing=not ledger_path.exists(),tags=str(args.tags),vectors=str(args.vectors),
                    vector_bytes=vector_bytes,complete_vector_rows=vector_rows,
                    ignored_vector_tail_bytes=vector_bytes%2048,ledger_rows=len(ledger),
                    tagged_unique_paths=len(tags),eligible_rows=len(eligible),excluded=dict(rejected),
                    pack_cap=args.pack_cap,selected=len(chosen),selection=stats,
                    counts={s:dict(Counter(r['split'] for r in chosen if r['stratum']==s)) for s in RULES})
    (HERE/'selection.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({k:metadata[k] for k in ['selected','eligible_rows','counts','selection']},indent=2))



def ledger_fallback_check():
    """Exercise complete pilot builds without models, audio, or a pre-existing metadata ledger."""
    import contextlib
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch
    with tempfile.TemporaryDirectory(prefix='uap-benchmark-test-') as temporary:
        root = Path(temporary)
        tags = root / 'AUDIO_TAGS.csv'
        tag_rows = [
            dict(relative_path='Music Pack/calm piano.wav', seconds_tagged='60', clap_row='0',
                 music='.9', silence='0', top_labels='Music 0.9'),
            dict(relative_path='Nature Pack/rain.wav', seconds_tagged='5', clap_row='1',
                 music='0', silence='0', top_labels='Rain 0.9'),
        ]
        write_csv(tags, tag_rows)
        vectors = root / 'AUDIO_TAGS_clap.f32'
        vectors.write_bytes(bytes(2 * 2048))
        args = SimpleNamespace(tags=tags, vectors=vectors, ledger=None,
                               seed=41, per_stratum=3, pack_cap=2)
        def pilot(name):
            out = root / name
            out.mkdir()
            with patch.dict(globals(), HERE=out), contextlib.redirect_stdout(io.StringIO()):
                build(args)
            return read_csv(out / 'snapshot.csv'), json.loads((out / 'selection.json').read_text(encoding='utf-8'))
        from library_metadata import _missing_ledger_noted
        _missing_ledger_noted.discard(str(root.resolve()))
        notes = io.StringIO()
        with contextlib.redirect_stderr(notes):
            rows, meta = pilot('without-ledger')
            pilot('without-ledger-again')
        assert notes.getvalue().count('NOTE') == 1 and 'AUDIO_LEDGER_files.csv' in notes.getvalue(), notes.getvalue()
        by_path = {row['relative_path']: row for row in rows}
        assert len(rows) == 2 and by_path['Music Pack/calm piano.wav']['pack'] == 'Music Pack', rows
        assert float(by_path['Music Pack/calm piano.wav']['duration_seconds']) == 60, rows
        assert meta['ledger_missing'] and meta['selection']['eligible_candidates']['calm_instrumental / positive_candidate'] == 1, meta
        fresh = root / 'AUDIO_LEDGER_files.csv'
        fields = ['relative_path','pack','duration_seconds','sample_rate','channels','size_bytes','mtime_ns']
        write_csv(fresh, [dict(relative_path=row['relative_path'],pack=row['relative_path'].split('/')[0],
                              duration_seconds=90 if i == 0 else 5,sample_rate=48000,channels=1,
                              size_bytes=100,mtime_ns=1) for i,row in enumerate(tag_rows)],fields)
        rows, meta = pilot('fresh-ledger')
        assert float(next(row for row in rows if row['relative_path'].startswith('Music Pack'))['duration_seconds']) == 90, rows
        assert not meta['ledger_missing'] and meta['selection']['eligible_candidates']['calm_instrumental / positive_candidate'] == 1, meta
        assert not candidate(dict(by_path['Music Pack/calm piano.wav'], bucket='sfx'), 'calm_instrumental', 'positive_candidate')
        custom = root / 'custom.csv'
        write_csv(custom, [dict(relative_path='Nature Pack/rain.wav',pack='Custom Pack',duration_seconds=12)])
        args.ledger = custom
        rows, _ = pilot('custom-ledger')
        assert len(rows) == 1 and rows[0]['pack'] == 'Custom Pack' and float(rows[0]['duration_seconds']) == 12, rows
        args.ledger = root / 'missing-custom.csv'
        try:
            pilot('missing-custom-ledger')
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('An explicit missing custom ledger must not silently select a different library')
        args.ledger = custom
        vectors.write_bytes(b'')
        rows, meta = pilot('no-valid-vectors')
        assert rows == [] and meta['selected'] == 0 and meta['excluded']['invalid_vector_or_duration'] == 1, (rows, meta)
    print('PASS bench_select ledger builds: note once, tag fallback, seven-column music nomination, custom subset, empty input')


def main():
    global HERE
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('check')
    b = sub.add_parser('build')
    b.add_argument('--ledger',type=Path,help='custom ledger; default beside --tags, with tag fallback if missing')
    b.add_argument('--tags',type=Path,default=ROOT/'AUDIO_TAGS.csv')
    b.add_argument('--vectors',type=Path,default=ROOT/'AUDIO_TAGS_clap.f32')
    b.add_argument('--seed',type=int,default=20261010)
    b.add_argument('--per-stratum',type=int,default=18)
    b.add_argument('--pack-cap',type=int,default=8)
    b.add_argument('--out',type=Path,default=HERE)
    a = p.parse_args()
    if a.cmd=='check': return check()
    if a.per_stratum < 3 or a.pack_cap < 1: p.error('per-stratum >= 3 and pack-cap >= 1 required')
    HERE = a.out.resolve(); HERE.mkdir(parents=True,exist_ok=True)
    build(a)


if __name__ == '__main__':
    main()
