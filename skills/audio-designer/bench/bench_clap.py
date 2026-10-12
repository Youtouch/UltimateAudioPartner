"""Offline CPU-only CLAP baseline against an append-safe snapshot."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent


def check():
    assert complete_vector_rows(4096+101) == (2,101)
    v = np.array([[1.,0.],[0.,1.]],dtype=np.float32)
    q = np.array([1.,0.],dtype=np.float32)
    assert np.allclose(cosine_scores(v,q), [1,0])
    assert np.allclose(cosine_scores(v*2,q*3), [1,0])
    try: cosine_scores(np.zeros((1,2)),q)
    except ValueError: pass
    else: assert False, 'zero vectors must fail'
    print('PASS bench_clap check: partial vector tail, cosine normalization, zero-vector rejection')


def complete_vector_rows(size):
    return divmod(size,512*4)


def cosine_scores(vectors,query):
    norms=np.linalg.norm(vectors,axis=1); qnorm=np.linalg.norm(query)
    if not np.isfinite(vectors).all() or not np.isfinite(query).all() or np.any(norms<=0) or qnorm<=0:
        raise ValueError('Nonfinite or zero CLAP vectors')
    return (vectors@query)/(norms*qnorm)


def run(args):
    # Explicit offline flags and a cached LOCAL path prevent accidental downloads.
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch, transformers
    from transformers import ClapTextModelWithProjection, AutoTokenizer
    from bench_select import read_csv, write_csv
    from bench_eval import evaluate, print_report, proxy_relevant, retrieval_metrics
    torch.set_num_threads(args.threads)
    start=time.perf_counter()
    metadata=json.loads((HERE/'selection.json').read_text(encoding='utf-8'))
    snapshot=read_csv(HERE/'snapshot.csv'); queries=read_csv(HERE/'queries.csv')
    benchmark=read_csv(HERE/'benchmark.csv')
    vocab={r['tag'] for r in read_csv(HERE/'vocabulary.csv')}
    vector_file=Path(metadata['vectors'])
    count,tail=complete_vector_rows(metadata['vector_bytes'])
    if vector_file.stat().st_size < metadata['vector_bytes']: raise ValueError('Vector file shrank after selection')
    mm=np.memmap(vector_file,mode='r',dtype='<f4',shape=(count,512))
    indices=np.array([int(r['clap_row']) for r in snapshot],dtype=np.int64)
    vectors=np.array(mm[indices],copy=True)
    vector_hash=hashlib.sha256(vectors.tobytes()).hexdigest()
    norms=np.linalg.norm(vectors,axis=1)
    paths=[r['relative_path'] for r in snapshot]
    print(f'Offline CLAP text encoder on CPU; {len(paths)} cached audio vectors; batch size 1',flush=True)
    # Full cached checkpoint contains both towers; Transformers loads only text tower parameters.
    tokenizer=AutoTokenizer.from_pretrained(str(args.cached_model),local_files_only=True)
    model=ClapTextModelWithProjection.from_pretrained(str(args.cached_model),local_files_only=True,
                                                    use_safetensors=(args.cached_model/'model.safetensors').exists()).to('cpu').eval()
    text_vectors=[]
    with torch.inference_mode():
        for i,q in enumerate(queries,1):
            inputs=tokenizer([q['query']],padding=True,return_tensors='pt')
            output=model(**inputs)
            features=output.text_embeds
            vec=torch.nn.functional.normalize(features,dim=-1).cpu().numpy()[0]
            text_vectors.append(vec)
            print(f"text {i}/{len(queries)}: {q['query_id']}",flush=True)
    text_vectors=np.array(text_vectors,dtype=np.float32)
    scores=np.stack([cosine_scores(vectors,q) for q in text_vectors],axis=1)
    rankings={}; csv_rows=[]; full_metrics={}; top100=[]
    pilot={r['relative_path'] for r in benchmark}
    for j,q in enumerate(queries):
        order=sorted(range(len(paths)),key=lambda i:(-float(scores[i,j]),paths[i]))
        ranking=[paths[i] for i in order]
        rankings[q['query_id']]=ranking
        selected=[i for i in order if paths[i] in pilot]
        csv_rows.extend(dict(query_id=q['query_id'],relative_path=paths[i],rank=n,
                             score=float(scores[i,j])) for n,i in enumerate(selected,1))
        positives={r['relative_path'] for r in snapshot if proxy_relevant(q,r)}
        full_metrics[q['query_id']]=retrieval_metrics(ranking,positives,len(paths))
        top100.extend(dict(query_id=q['query_id'],rank=n,relative_path=paths[i],score=float(scores[i,j]),
                           filename_proxy_relevant=int(paths[i] in positives),
                           ast_silence=snapshot[i].get('silence',''),duration_seconds=snapshot[i]['duration_seconds'])
                      for n,i in enumerate(order[:100],1))
    np.savez_compressed(HERE/'clap_baseline_scores.npz',paths=np.array(paths),query_ids=np.array([q['query_id'] for q in queries]),
                        scores=scores,text_vectors=text_vectors)
    write_csv(HERE/'clap_rankings.csv',csv_rows)
    write_csv(HERE/'clap_library_top100.csv',top100)
    report=evaluate(benchmark,queries,vocab,rankings,mode='proxy')
    (HERE/'clap_proxy.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print_report(report)
    provenance=dict(mode=report['mode'],scope='full tagged snapshot; no human truth',per_query=full_metrics,
                    audio_vectors_sha256=vector_hash,cached_model=str(args.cached_model),
                    model_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.cached_model.iterdir()
                                 if p.is_file() and p.suffix in {'.json','.txt'}},
                    model_checkpoint_bytes=(args.cached_model/('model.safetensors' if (args.cached_model/'model.safetensors').exists() else 'pytorch_model.bin')).stat().st_size,
                    python=__import__('sys').version,torch=torch.__version__,transformers=transformers.__version__,
                    device='cpu',text_batch_size=1,audio_vectors=len(paths),vector_norm_min=float(norms.min()),
                    vector_norm_max=float(norms.max()),vector_tail_bytes_ignored=tail,elapsed_seconds=time.perf_counter()-start,
                    queries_sha256=hashlib.sha256((HERE/'queries.csv').read_bytes()).hexdigest(),
                    snapshot_sha256=hashlib.sha256((HERE/'snapshot.csv').read_bytes()).hexdigest())
    (HERE/'clap_library_proxy.json').write_text(json.dumps(provenance,indent=2,allow_nan=False),encoding='utf-8')
    print(f"Saved CPU baseline in {provenance['elapsed_seconds']:.2f}s",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True); sub.add_parser('check')
    r=sub.add_parser('run')
    r.add_argument('--cached-model',type=Path,default=Path.home()/'.cache/huggingface/hub/models--laion--clap-htsat-unfused/snapshots/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a')
    r.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if a.cmd=='check': return check()
    if a.threads<1: p.error('threads must be >= 1')
    return run(a)


if __name__ == '__main__':
    main()
