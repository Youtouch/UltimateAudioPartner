"""Retrieval and multilabel evaluation with explicit reviewed/proxy contracts."""
import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
PROXY = 'PROXY - filename truth, biased toward name-based methods'


def check():
    m = retrieval_metrics(['b','a','c'], {'a','c'}, 3)
    assert abs(m['AP'] - (0.5 + 2/3)/2) < 1e-12
    assert m['P@10'] == .2 and m['P@50'] == .04
    assert retrieval_metrics(['b'], {'a'}, 2)['AP'] == 0
    assert retrieval_metrics([], set(), 0)['AP'] is None
    assert dedup_paths(['a','a','b']) == ['a','b']
    v = {'click','speech','piano','music','sung_voice','choir'}
    base = dict(relative_path='a',group_id='g',split='dev',sources='click',uncertain='n',reviewed_by='AB')
    assert reviewed_tags(base,v) == {'click'}
    assert reviewed_tags(dict(base,uncertain='y'),v) is None
    assert reviewed_tags(dict(base,reviewed_by=''),v) is None
    assert reviewed_tags(dict(base,sources=''),v) == set()
    q = dict(relevant_tags='piano;music',match='all',exclude_tags='speech;sung_voice;choir',
             min_duration='',max_duration='')
    assert relevant(q, {'piano','music'}, 3)
    assert not relevant(q, {'piano','music','choir'}, 3)
    assert not relevant(q, {'piano'}, 3)
    c = counts_metrics(1,1,1)
    assert c['precision'] == .5 and c['recall'] == .5 and c['F1'] == .5
    assert counts_metrics(0,0,0)['F1'] is None
    print('PASS bench_eval check: hand-derived AP/P@k, incomplete ranking, unknown truth, exclusions, PRF')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def tags(value):
    return {t.strip() for t in (value or '').split(';') if t.strip()}


def dedup_paths(paths):
    return list(dict.fromkeys(paths))


def reviewed_tags(row, vocabulary):
    present = tags(row.get('sources',''))
    if present-vocabulary: raise ValueError('Unknown source tags: '+str(present-vocabulary))
    state = row.get('uncertain','').strip().lower()
    if state not in {'','y','n'}: raise ValueError('uncertain must be empty, y or n')
    if not row.get('reviewed_by','').strip() or state != 'n': return None
    return present


def duration_ok(query, duration):
    return ((not query.get('min_duration') or duration >= float(query['min_duration'])) and
            (not query.get('max_duration') or duration < float(query['max_duration'])))


def relevant(query, present, duration):
    wanted = tags(query['relevant_tags'])
    match = query.get('match','any') or 'any'
    if match not in {'any','all'} or not wanted: raise ValueError('query requires nonempty tags and any/all match')
    yes = bool(wanted & present) if match == 'any' else wanted <= present
    return yes and not (tags(query.get('exclude_tags','')) & present) and duration_ok(query,duration)


def proxy_relevant(query,row):
    name = Path(row['relative_path'].replace('\\','/')).name
    return (bool(re.search(query['proxy_regex'],name,re.I)) and
            not (query.get('proxy_exclude_regex') and re.search(query['proxy_exclude_regex'],name,re.I)) and
            duration_ok(query,float(row['duration_seconds'])))


def retrieval_metrics(ranking, positives, universe_size):
    ranking = dedup_paths(ranking)
    hits = 0; ap = 0.
    for i,path in enumerate(ranking,1):
        if path in positives: hits+=1; ap+=hits/i
    return {'P@10':sum(p in positives for p in ranking[:10])/10,
            'P@50':sum(p in positives for p in ranking[:50])/50,
            'AP':ap/len(positives) if positives else None,
            'relevant':len(positives),'returned':len(ranking),'eligible':universe_size,
            'hits@10':sum(p in positives for p in ranking[:10]),
            'hits@50':sum(p in positives for p in ranking[:50])}


def counts_metrics(tp,fp,fn):
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    return dict(TP=tp,FP=fp,FN=fn,support=tp+fn,precision=precision,recall=recall,
                F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def mean(values):
    values = [x for x in values if x is not None]
    return sum(values)/len(values) if values else None


def load_rankings(path, queries):
    qids={q['query_id'] for q in queries}; result=defaultdict(list); positions=defaultdict(set)
    for r in read_csv(path):
        qid=r['query_id']
        if qid not in qids: raise ValueError('Unknown query_id: '+qid)
        rank=int(r['rank'])
        if rank < 1 or rank in positions[qid]: raise ValueError('Ranks must be positive and unique per query')
        positions[qid].add(rank)
        result[qid].append((rank,r['relative_path'].replace('\\','/')))
    return {q:dedup_paths(p for _,p in sorted(items)) for q,items in result.items()}


def callable_rankings(spec,queries,rows):
    filename,function=spec.rsplit(':',1)
    module_spec=importlib.util.spec_from_file_location('benchmark_ranking_plugin',filename)
    module=importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    rank=getattr(module,function)
    result={}
    for q in queries:
        # Ground truth, split IDs and nomination strata must not be offered as ranking features.
        candidates=[{k:r[k] for k in ('relative_path','duration_seconds') if k in r} for r in rows]
        output=list(rank(dict(q),candidates))
        result[q['query_id']]=dedup_paths((x['relative_path'] if isinstance(x,dict) else str(x)).replace('\\','/') for x in output)
    return result


def tag_report(rows, vocabulary, score_rows, threshold, mode):
    if mode=='proxy':
        return dict(status='not_evaluated: source truth is unreviewed; proxy regexes define QUERY relevance only')
    if not score_rows: return dict(status='not_evaluated: no full tag-score CSV supplied')
    truth={r['relative_path']:reviewed_tags(r,vocabulary) for r in rows}
    truth={p:t for p,t in truth.items() if t is not None}
    scores={}
    for r in score_rows:
        tag=r['tag']; value=float(r['score']); key=(r['relative_path'].replace('\\','/'),tag)
        if tag not in vocabulary or not math.isfinite(value): raise ValueError('Invalid tag or nonfinite score')
        if key in scores: raise ValueError('Duplicate tag prediction '+str(key))
        scores[key]=value
    by_tag={}
    for tag in sorted(vocabulary):
        available=sum((p,tag) in scores for p in truth)
        if available != len(truth) or not truth:
            by_tag[tag]=dict(status='incomplete_scores',scored=available,eligible=len(truth)); continue
        tp=fp=fn=0
        for path,present in truth.items():
            pred=scores[path,tag]>=threshold; actual=tag in present
            tp+=pred and actual; fp+=pred and not actual; fn+=not pred and actual
        by_tag[tag]=dict(status='ok',**counts_metrics(tp,fp,fn))
    valid=[r for r in by_tag.values() if r['status']=='ok']
    return dict(status='ok',threshold=threshold,per_tag=by_tag,
                macro={k:mean(r[k] for r in valid) for k in ('precision','recall','F1')},
                macro_supported={k:mean(r[k] for r in valid if r['support']>0) for k in ('precision','recall','F1')},
                micro=counts_metrics(*(sum(r[k] for r in valid) for k in ('TP','FP','FN'))),
                complete_tags=len(valid),total_tags=len(vocabulary))


def evaluate(rows,queries,vocabulary,rankings,mode='reviewed',score_rows=None,threshold=.5):
    if mode not in {'reviewed','proxy'}: raise ValueError('mode must be reviewed or proxy')
    if len({r['relative_path'] for r in rows})!=len(rows): raise ValueError('Duplicate benchmark paths')
    if len({q['query_id'] for q in queries})!=len(queries): raise ValueError('Duplicate query IDs')
    groups=defaultdict(set)
    for r in rows:
        if r['split'] not in {'dev','test'}: raise ValueError('split must be dev or test')
        groups[r['group_id']].add(r['split'])
        reviewed_tags(r,vocabulary)
    if any(len(s)>1 for s in groups.values()): raise ValueError('Group leakage across splits')
    for q in queries:
        if (tags(q['relevant_tags'])|tags(q.get('exclude_tags','')))-vocabulary: raise ValueError('Unknown query tags')
        relevant(q,set(),1.)
        if mode=='proxy' and not q.get('proxy_regex'): raise ValueError('Proxy needs explicit filename regex')
    report=dict(mode=PROXY if mode=='proxy' else 'REVIEWED - fixed excerpt truth',
                metrics_contract='P@k denominator is k; missing results are misses; AP denominator is all known relevant; unjudged rows condensed out',
                splits={})
    for split in ('dev','test'):
        subset=[r for r in rows if r['split']==split]
        eligible=[r for r in subset if mode=='proxy' or reviewed_tags(r,vocabulary) is not None]
        paths={r['relative_path'] for r in eligible}
        per_query={}
        for q in queries:
            positive={r['relative_path'] for r in eligible if (proxy_relevant(q,r) if mode=='proxy' else
                      relevant(q,reviewed_tags(r,vocabulary),float(r['duration_seconds'])))}
            full=rankings.get(q['query_id'],[])
            ranking=dedup_paths(p for p in full if p in paths)
            metrics=retrieval_metrics(ranking,positive,len(paths))
            metrics.update(query=q['query'],coverage=len(ranking)/len(paths) if paths else None,
                           status='ok' if paths else 'no_reviewed_rows')
            if not paths: metrics.update({'P@10':None,'P@50':None})
            per_query[q['query_id']]=metrics
        qs=list(per_query.values()); support=sum(r['relevant'] for r in qs)
        report['splits'][split]=dict(rows=len(subset),eligible=len(eligible),excluded_unknown=len(subset)-len(eligible),
            per_query=per_query,macro={k:mean(r[k] for r in qs) for k in ('P@10','P@50','AP')},
            micro={'P@10':sum(r['hits@10'] for r in qs)/(10*len(qs)) if paths and qs else None,
                   'P@50':sum(r['hits@50'] for r in qs)/(50*len(qs)) if paths and qs else None,
                   'AP_support_weighted':sum((r['AP'] or 0)*r['relevant'] for r in qs)/support if support else None},
            queries_with_positive_support=sum(r['relevant']>0 for r in qs),
            tag_metrics=tag_report(subset,vocabulary,score_rows,threshold,mode))
    return report


def print_report(report):
    print(report['mode'])
    for split,s in report['splits'].items():
        print(f"{split}: {s['eligible']}/{s['rows']} eligible; macro {s['macro']}; micro {s['micro']}")
        print('query_id                 P@10      P@50      AP        relevant/returned')
        for q,r in s['per_query'].items():
            fmt=lambda x: 'NA' if x is None else f'{x:.4f}'
            print(f"{q:24} {fmt(r['P@10']):9} {fmt(r['P@50']):9} {fmt(r['AP']):9} {r['relevant']}/{r['returned']}")
        print('tag metrics:',s['tag_metrics']['status'])
        tm=s['tag_metrics']
        if tm['status']=='ok':
            print('tag macro:',tm['macro'],'tag micro:',tm['micro'])
            for tag,r in tm['per_tag'].items():
                if r['status']=='ok': print(tag, {k:r[k] for k in ('precision','recall','F1','support')})


def integration_check():
    v={'click','speech'}
    rows=[dict(relative_path=p,group_id=g,split=s,duration_seconds='1',sources=t,uncertain='n',reviewed_by='AB')
          for p,g,s,t in [('a','g1','dev','click'),('b','g2','dev',''),('c','g3','test','click')]]
    q=[dict(query_id='q',query='click',relevant_tags='click',match='any',exclude_tags='',proxy_regex='a')]
    scores=[dict(relative_path=p,tag=t,score=str(x)) for p,t,x in
            [('a','click',.9),('b','click',.8),('c','click',.1),('a','speech',.1),('b','speech',.1),('c','speech',.1)]]
    r=evaluate(rows,q,v,{'q':['b','a','c']},score_rows=scores)
    assert r['splits']['dev']['per_query']['q']['AP']==.5
    assert r['splits']['test']['per_query']['q']['AP']==1
    assert r['splits']['dev']['tag_metrics']['per_tag']['click']['precision']==.5
    assert r['splits']['test']['tag_metrics']['per_tag']['click']['recall']==0
    rows[0]['uncertain']='y'
    assert evaluate(rows,q,v,{'q':['a']})['splits']['dev']['excluded_unknown']==1
    # A missing score must not silently become zero or a false negative.
    partial=tag_report(rows,v,scores[:1],.5,'reviewed')
    assert partial['per_tag']['click']['status']=='incomplete_scores'
    # CSV rank order matters even when scores disagree; duplicates collapse at first rank.
    import tempfile
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        fixture=Path(directory)/'ranking.csv'
        fixture.write_text('query_id,relative_path,rank,score\nq,a,3,9\nq,b,1,0\nq,b,2,8\n',encoding='utf-8')
        assert load_rankings(fixture,q)=={'q':['b','a']}
        plugin=Path(directory)/'fixture_plugin.py'
        plugin.write_text('def rank(query, rows):\n    assert query["query_id"] == "q"\n    assert all(set(row) == {"relative_path", "duration_seconds"} for row in rows)\n    return [row["relative_path"] for row in reversed(rows)]\n',encoding='utf-8')
        assert callable_rankings(str(plugin)+':rank',q,rows)=={'q':['c','b','a']}
        fixture.write_text('query_id,relative_path,rank\nq,a,1\nq,b,1\n',encoding='utf-8')
        try: load_rankings(fixture,q)
        except ValueError: pass
        else: assert False,'ambiguous CSV ranks must fail'
    rows[2]['group_id']='g2'
    try: evaluate(rows,q,v,{})
    except ValueError: pass
    else: assert False,'leakage must fail'
    print('PASS bench_eval integration: separate splits, multilabel metrics, uncertain exclusion, leakage rejection')


def main():
    p=argparse.ArgumentParser(
        description='Score retrieval and label predictions against reviewed or filename proxy truth.',
        epilog='Example: python bench/bench_eval.py run --rankings rankings.csv --mode reviewed',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub=p.add_subparsers(dest='cmd',required=True, title='commands', help='available commands')
    sub.add_parser(
        'check',
        description='Run synthetic checks without a library or model download.',
        epilog='Example: python bench/bench_eval.py check',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help='Run synthetic checks without a library or model download.',
    )
    e=sub.add_parser(
        'run',
        description='Evaluate rankings with explicit reviewed or proxy truth and coverage.',
        epilog='Example: python bench/bench_eval.py run --rankings rankings.csv --mode reviewed --out results.json',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help='Evaluate rankings with explicit reviewed or proxy truth and coverage.',
    )
    e.add_argument('--benchmark',type=Path,default=HERE/'benchmark.csv', help='review-sheet CSV defining candidates and truth')
    e.add_argument('--queries',type=Path,default=HERE/'queries.csv', help='frozen query-contract CSV')
    e.add_argument('--vocabulary',type=Path,default=HERE/'vocabulary.csv', help='canonical-tag vocabulary CSV')
    method=e.add_mutually_exclusive_group(required=True)
    method.add_argument('--rankings',type=Path, help='CSV of query_id, relative_path and positive integer rank')
    method.add_argument('--callable',help='local_file.py:function; function(query_dict,candidate_rows) returns ranked relative paths')
    e.add_argument('--tag-scores',type=Path, help='optional full-score CSV with relative_path, tag and score')
    e.add_argument('--threshold',type=float,default=.5, help='label score at or above which a prediction is positive')
    e.add_argument('--mode',choices=['reviewed','proxy'],default='reviewed', help='reviewed listener truth or explicitly biased filename proxy truth')
    e.add_argument('--out',type=Path, help='optional JSON report path; metrics also print to stdout')
    a=p.parse_args()
    if a.cmd=='check': check(); return integration_check()
    if not math.isfinite(a.threshold): p.error('threshold must be finite')
    rows=read_csv(a.benchmark); queries=read_csv(a.queries)
    vocabulary={r['tag'] for r in read_csv(a.vocabulary)}
    rankings=load_rankings(a.rankings,queries) if a.rankings else callable_rankings(a.callable,queries,rows)
    report=evaluate(rows,queries,vocabulary,rankings,a.mode,read_csv(a.tag_scores) if a.tag_scores else None,a.threshold)
    print_report(report)
    if a.out: a.out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__ == '__main__':
    main()
