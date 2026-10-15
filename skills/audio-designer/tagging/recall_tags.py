"""Offline category catalogue and retrieval. Sources are candidates, never listening truth."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SKILL = HERE.parent
sys.path.insert(0, str(SKILL / 'scripts'))
from library_metadata import read_metadata, relative_path

SCHEMA = 'uap-recall-v1'
WARNING = 'Names and model outputs are incomplete candidate evidence; precision has not been established by listening.'





_CACHE = {}

def taxonomy_module():
    import ucs
    return ucs


def read_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def source_stamps(root):
    result={}
    for name in ('AUDIO_TAGS.csv','AUDIO_TAGS_clap.f32','AUDIO_TAGS_clap.json','AUDIO_LEDGER_files.csv','AUDIO_GROUPS.csv'):
        path=Path(root)/name
        try:stat=path.stat()
        except OSError:stat=None
        result[name]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns) if stat else None
    return result


def audio_source_stamps(root,paths):
    """Aligned actual audio identity, including missing sources explicitly."""
    sizes,mtimes=[],[]
    for path in paths:
        try:stat=(Path(root)/str(path)).stat()
        except OSError:stat=None
        sizes.append(stat.st_size if stat else -1);mtimes.append(stat.st_mtime_ns if stat else -1)
    return np.array(sizes,np.int64),np.array(mtimes,np.int64)


def provenance_status(root,manifest):
    expected=manifest.get('source_stamps')
    if expected is None:return ['Catalogue predates source freshness stamps; calibration freshness cannot be verified.']
    actual=source_stamps(root)
    return [name+' changed since fit' for name,stamp in expected.items() if actual.get(name)!=stamp]


def number(value, default=float('nan')):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def integer(value):
    try:return int(value or 0)
    except (ValueError,TypeError):return 0


def load_library(root):
    """Read-only loader: every ledger path survives, including missing embeddings."""
    root = Path(root)
    tags = {relative_path(r['relative_path']): r for r in read_rows(root / 'AUDIO_TAGS.csv')}
    meta = read_metadata(root, tags)
    paths = sorted(meta)
    seconds = np.array([number(meta[p].get('duration_seconds')) for p in paths])
    packs = [meta[p].get('pack') or (p.split('/')[0] if '/' in p else '__root__') for p in paths]
    raw_path = root / 'AUDIO_TAGS_clap.f32'
    rows = raw_path.stat().st_size // (512 * 4) if raw_path.exists() else 0
    x = np.zeros((len(paths), 512), dtype=np.float32)
    valid = np.zeros(len(paths), bool)
    tag_source_valid=np.ones(len(paths),bool)
    mapped = np.memmap(raw_path, mode='r', dtype='<f4', shape=(rows, 512)) if rows else None
    for i, path in enumerate(paths):
        row=tags.get(path,{})
        size,mtime=integer(row.get('size_bytes')),integer(row.get('mtime_ns'))
        if size>0 and mtime>0:
            try:
                stat=(root/path).stat()
                tag_source_valid[i]=(stat.st_size,stat.st_mtime_ns)==(size,mtime)
            except OSError:tag_source_valid[i]=False
        try:
            index = int(tags.get(path, {}).get('clap_row', -1))
        except (ValueError, TypeError):
            continue
        if 0 <= index < rows:
            x[i] = mapped[index]
            valid[i] = tag_source_valid[i] and np.isfinite(x[i]).all() and np.linalg.norm(x[i]) > 0
    norm = np.linalg.norm(x[valid], axis=1)
    x[valid] /= norm[:, None]
    x[~valid] = 0
    return dict(paths=paths, packs=packs, seconds=seconds, tags=tags, meta=meta,
                embeddings=x, valid=valid, tag_source_valid=tag_source_valid,root=root)


def label_scores(library, taxonomy):
    # Printed labels are capped and rounded; an absent label is unknown.
    pair = re.compile(r'(.+?) (\d+(?:\.\d+)?)(?:; |, |$)')
    wanted = defaultdict(list)
    for j, row in enumerate(taxonomy):
        for label in row.get('audioset_labels', []):
            for variant in taxonomy_module().label_variants(label):
                wanted[variant.casefold()].append(j)
    scores = np.zeros((len(library['paths']), len(taxonomy)), np.float32)
    tag_valid=library.get('tag_source_valid')
    for i, path in enumerate(library['paths']):
        if tag_valid is not None and not tag_valid[i]:continue
        parsed = pair.findall(library['tags'].get(path, {}).get('top_labels', ''))
        for label, value in parsed:
            for j in wanted.get(label.casefold(), ()):
                scores[i, j] = max(scores[i, j], float(value))
    return scores


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='\n',prefix='.uap-recall-',suffix='.tmp',dir=path.parent,delete=False) as target:
            temporary=Path(target.name)
            json.dump(value,target,indent=2,ensure_ascii=False,allow_nan=False);target.write('\n')
            target.flush();os.fsync(target.fileno())
        os.link(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def immutable_npz(path, **arrays):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',prefix='.uap-recall-',suffix='.tmp',dir=path.parent,delete=False) as target:
            temporary=Path(target.name)
            np.savez_compressed(target,**arrays);target.flush();os.fsync(target.fileno())
        os.link(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def load_catalog(root, manifest_path=None):
    root=Path(root)
    candidates=[Path(manifest_path)] if manifest_path else list(root.glob('AUDIO_RECALL_*.json'))
    if not manifest_path and (root/'AUDIO_RECALL.json').exists():
        candidates.append(root/'AUDIO_RECALL.json')
    for path in sorted(candidates,key=lambda p:p.stat().st_mtime_ns,reverse=True):
        key=(str(path.resolve()),path.stat().st_mtime_ns)
        if key in _CACHE:
            manifest,data,name=_CACHE[key]
            return ({**manifest,'stale_reasons':provenance_status(root,manifest)},data,name)
        try:
            manifest=json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError,json.JSONDecodeError):
            if manifest_path:raise
            continue  # Interrupted new metadata must not hide an older complete run.
        if manifest.get('schema') != SCHEMA:
            continue
        arrays=root/manifest['arrays']
        if arrays.parent.resolve()!=root.resolve():
            raise ValueError('Catalogue arrays must be in library root')
        try:
            with np.load(arrays,allow_pickle=False) as data:
                value=(manifest,{k:data[k] for k in data.files},path.name)
        except (OSError,ValueError):
            if manifest_path:raise
            continue
        _CACHE.clear();_CACHE[key]=value
        return ({**manifest,'stale_reasons':provenance_status(root,manifest)},value[1],value[2])
    return None


def resolve(query, taxonomy, aliases=None):
    catid={r['catid']:r for r in taxonomy}
    if query in catid:
        row=catid[query]
        return dict(catids=[query],label=row['category']+' › '+row['subcategory'],remaining_query='')
    return taxonomy_module().resolve_query(query,aliases=aliases) if aliases else taxonomy_module().resolve_query(query)

def build(root, out=None, prompts=False, progress=print):
    """Build source candidates only. No fitting, benchmark or model cache is required."""
    if prompts:
        raise ValueError('Prompt extraction is local research; build uses existing tag evidence.')
    root=Path(root); taxonomy=taxonomy_module().load_taxonomy(); library=load_library(root)
    paths=library['paths']; n=len(paths); c=len(taxonomy)
    names=np.zeros((n,c),bool); index={r['catid']:j for j,r in enumerate(taxonomy)}
    for i,path in enumerate(paths):
        for catid in taxonomy_module().weak_matches(path): names[i,index[catid]]=True
    labels=label_scores(library,taxonomy)
    sizes,mtimes=audio_source_stamps(root,paths)
    out=Path(out) if out else root/('AUDIO_RECALL_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
    arrays=out.with_suffix('.npz')
    if out.exists() or arrays.exists(): raise FileExistsError(out)
    report=dict(schema=SCHEMA,arrays=arrays.name,files=n,source_stamps=source_stamps(root),
                prompt_info={'available':False},coverage={'calibrated':False},
                warning=WARNING,source_policy='Existing filename references and model labels; uncalibrated.')
    immutable_npz(arrays,paths=np.array(paths),catids=np.array([r['catid'] for r in taxonomy]),
                  name=names,label=labels>=.2,labels=labels,source_sizes=sizes,source_mtimes=mtimes)
    immutable_json(out,report); progress(f'Catalogue: {n} files, uncalibrated source candidates')
    _CACHE.clear()
    return out,report


def query_catalog(root, query, precision=.9, manifest_path=None):
    if precision not in (.8,.9): raise ValueError('Supported display targets: 0.8 or 0.9')
    ucs=taxonomy_module(); taxonomy=ucs.load_taxonomy(); coverage=resolve(query,taxonomy,ucs.read_aliases(root))
    coverage.update(precision=precision,calibrated=False,warning=WARNING,deployment_verified=False,
                    adoption_reason='Listening precision has not been established. Candidates remain uncalibrated.')
    loaded=load_catalog(root,manifest_path)
    if not loaded:
        coverage['reason']='No category catalogue. Run tagging/recall_tags.py build first.'
        return dict(results=[],coverage=coverage,counts=dict(total=0,confirmed=0,likely=0),catalog=None)
    manifest,data,catalog=loaded; ids=data['catids'].tolist(); paths=data['paths'].tolist()
    requested=[ids.index(c) for c in coverage.get('catids',[]) if c in ids]; stale=manifest.get('stale_reasons',[])
    merged={}
    # Old research manifests are readable, but experimental predictions and gate verdicts
    # cannot create runtime confidence. Direct source evidence stays available.
    for j in requested:
        for i in np.flatnonzero(data['name'][:,j]|data['label'][:,j]):
            path=paths[i]; sources=[s for s in ('name','label') if data[s][i,j]]
            file_stale=False
            if 'source_sizes' in data:
                try:
                    stat=(Path(root)/path).stat()
                    file_stale=(stat.st_size,stat.st_mtime_ns)!=(int(data['source_sizes'][i]),int(data['source_mtimes'][i]))
                except OSError: file_stale=True
            if path not in merged:
                merged[path]=dict(path=path,catids=[],sources=[],confidence='likely',calibrated=False,
                    adopted=False,score=None,threshold=None,calibration_qualified=False,
                    independent_source_agreement=False,stale=bool(stale) or file_stale,
                    precision_basis='uncalibrated source candidate')
            row=merged[path];row['catids'].append(ids[j]);row['sources']=sorted(set(row['sources']+sources))
    results=list(merged.values())
    from audio_describe import split_query,read_descriptors,table_for,word_mask
    _,filters=split_query(coverage.get('remaining_query',''))
    if filters and results:
        table=table_for([r['path'] for r in results],read_descriptors(root)); match=np.ones(len(results),bool)
        for word in filters:match &= word_mask(word,table)
        for row,yes in zip(results,match):row['matches_words']=bool(yes)
    results.sort(key=lambda r:(not r.get('matches_words',True),r['path']))
    coverage.update(stale=bool(stale) or any(r['stale'] for r in results),stale_reasons=stale,
                    unsupported_catids=coverage.get('catids',[]),calibrated_catids=[],calibration_qualified_catids=[],
                    threshold_policy='All direct source candidates; no top K; no calibrated precision claim.')
    return dict(results=results,coverage=coverage,counts=dict(total=len(results),confirmed=0,likely=len(results)),catalog=catalog)


def category_tree(root):
    taxonomy=taxonomy_module().load_taxonomy();loaded=load_catalog(root);counts={};parents={}
    if loaded:
        _,data,_=loaded; membership=data['name']|data['label'];ids=data['catids'].tolist()
        counts=dict(zip(ids,membership.sum(axis=0).tolist()))
        categories={r['catid']:r['category'] for r in taxonomy}
        for parent in set(categories.values()):
            selected=[j for j,c in enumerate(ids) if categories.get(c)==parent]
            parents[parent]=int(np.any(membership[:,selected],axis=1).sum()) if selected else 0
    return [dict(category=r['category'],subcategory=r['subcategory'],catid=r['catid'],count=counts.get(r['catid'],0),parent_count=parents.get(r['category'],0)) for r in taxonomy]


def boundary_sample(root,catid,n=20):
    if catid not in {r['catid'] for r in taxonomy_module().load_taxonomy()}: raise ValueError('Unknown CatID')
    loaded=load_catalog(root)
    if not loaded:return []
    _,data,catalog=loaded;ids=data['catids'].tolist()
    if catid not in ids:return []
    j=ids.index(catid);mask=data['name'][:,j]|data['label'][:,j]
    if 'splits' in data:mask &= data['splits']=='train'
    selected=np.flatnonzero(mask)
    # Disagreements first, deterministic and pack-spread; never claims a fitted boundary.
    selected=sorted(selected,key=lambda i:(data['name'][i,j]==data['label'][i,j],str(data['paths'][i])))[:max(0,min(int(n),20))]
    return [dict(path=str(data['paths'][i]),catid=catid,score=None,threshold=None,calibrated=False,
                 catalog=catalog,split='train' if 'splits' in data else 'unassigned',note='Source candidate check; targeted sample, not overall precision') for i in selected]


def main():
    parser=argparse.ArgumentParser(description='Build and search offline UCS category catalogues.',epilog='Example: python tagging/recall_tags.py build --root ./sounds')
    sub=parser.add_subparsers(dest='command',required=True,help='Catalogue operation')
    for name,description in [('build','Build a new uncalibrated catalogue from existing metadata.'),('query','Return every category candidate.'),('check','Run model-free catalogue regressions.')]:
        p=sub.add_parser(name,description=description,help=description,epilog='Example: python tagging/recall_tags.py '+name+(' --root ./sounds'+(' click' if name=='query' else '') if name!='check' else ''))
        if name!='check':p.add_argument('--root',required=True,help='Library root')
        if name=='build':p.add_argument('--out',help='New immutable manifest path, beside its arrays')
        if name=='query':
            p.add_argument('query',help='Category, family, synonym or CatID');p.add_argument('--precision',type=float,default=.9,choices=(.8,.9),help='Display target; candidates remain uncalibrated')
    args=parser.parse_args()
    if args.command=='build':print(build(args.root,args.out)[0])
    elif args.command=='query':print(json.dumps(query_catalog(args.root,args.query,args.precision),indent=2))
    else:
        import recall_tests
        recall_tests.check()


if __name__=='__main__':main()
